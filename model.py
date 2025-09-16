import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy


class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])
        self.class_colors = {
            (255, 255, 255): 0, # noise
            (160, 160, 160): 1, # lte
            (80, 80, 80): 2,    # 5g
            (0, 0, 0): 3        # radar
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()
 
        return image, label_mask

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0  
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)       
        optimizer.zero_grad()
        outputs = model(images)    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item() * images.size(0)      
        preds = torch.argmax(outputs, dim=1)     
        accuracy_metric(preds, labels)
        iou_metric(preds, labels)
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
        }) 
    epoch_loss = running_loss / len(dataloader.dataset)  
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
   
    return epoch_loss, mean_accuracy, mean_iou

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0    
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric(preds, labels)
            iou_metric(preds, labels)
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    
    return epoch_loss, mean_accuracy, mean_iou

class DSSL(nn.Module):
    """ Dynamic Selective Spectrum Layer (DSSL) Module """
    def __init__(self, channels=96, kernel_size=3, stride=1, dilations=[2, 4, 8]):
        super(DSSL, self).__init__()

        self.stride = stride
        self.dilations = dilations
        self.num_dilated_paths = len(dilations)
        self.total_num_paths = self.num_dilated_paths   # dilated + low + high

        # --- Dilated conv branches ---
        self.conv_paths = nn.ModuleList()
        for dilation in self.dilations:
            padding = dilation
            self.conv_paths.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels*2, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=channels, bias=False),
                    nn.BatchNorm2d(channels*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels*2, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )

        # --- Low-frequency branch ---
        self.low_freq_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # --- High-frequency branch ---
        self.high_freq_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # --- Attention mechanism ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.attention_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, self.total_num_paths * channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape

        # --- Dilated branches ---
        dilated_outputs = [conv(x) for conv in self.conv_paths]  # List of (B, C, H, W)

        # --- Low-frequency branch ---
        low_freq_raw = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low_freq = self.low_freq_proj(low_freq_raw)

        # --- High-frequency branch ---
        high_freq_raw = x - low_freq_raw
        high_freq = self.high_freq_proj(high_freq_raw)

        all_paths = dilated_outputs                     # length = total_num_paths
        stacked_paths = torch.stack(all_paths, dim=1)   # (B, num_paths, C, H, W)

        context_low = self.global_pool(low_freq)        # (B, C, 1, 1)
        context_high = self.global_pool(high_freq)      # (B, C, 1, 1)
        attention_input = torch.cat([context_low, context_high], dim=1)  # (B, 2C, 1, 1)

        gate_scores = self.attention_mlp(attention_input)                # (B, num_paths * C, 1, 1)
        gates = gate_scores.view(B, self.total_num_paths, C, 1, 1)
        gates = self.sigmoid(gates)

        # --- Weighted sum of all branches ---
        weighted_sum = torch.sum(stacked_paths * gates, dim=1)  # (B, C, H, W)

        return weighted_sum
        
class PSL(nn.Module):
    """ Pathwise Spatial Layer (PSL) Module """
    def __init__(self, in_channels, out_channels=32):
        super(PSL, self).__init__()
        concat_channels = out_channels * 2
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv3_cat = nn.Conv2d(concat_channels, concat_channels, kernel_size=3, padding=1, bias=False)
        self.bn_cat1 = nn.BatchNorm2d(concat_channels)
        self.depthwise = nn.Conv2d(concat_channels, concat_channels, kernel_size=3, padding=1, groups=concat_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(concat_channels)
        self.pointwise = nn.Conv2d(concat_channels, concat_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(concat_channels)

    def forward(self, x):
        out5 = self.bn5(self.conv5(x))
        out3 = self.bn3(self.conv3(x))
        concat = torch.cat([out5, out3], dim=1)
        concat = self.relu(concat)
        residual = concat
        main_path = self.relu(self.bn_cat1(self.conv3_cat(concat)))
        dw = self.relu(self.bn_dw(self.depthwise(main_path)))
        pw = self.relu(self.bn_pw(self.pointwise(dw)))
        out = pw + residual
        return out

class ASD(nn.Module):
    """ Adaptive Spectrum Distiller (ASD) Module"""
    def __init__(self, channels):
        super(ASD, self).__init__()
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        # GAP and Conv1x1
        self.conv1x1_avg_branch = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_avg_branch = nn.BatchNorm2d(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # GMP and Conv1x1
        self.conv1x1_max_branch = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_max_branch = nn.BatchNorm2d(channels)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv3 = nn.Conv2d(channels,96,kernel_size=1,padding=0, bias=False)
        self.bn = nn.BatchNorm2d(96)

    def forward(self, x):
        avg_branch_out = self.relu(self.bn_avg_branch(self.conv1x1_avg_branch(x)))
        avg_pooled_features = self.avg_pool(avg_branch_out)
        max_branch_out = self.relu(self.bn_max_branch(self.conv1x1_max_branch(x)))
        max_pooled_features = self.max_pool(max_branch_out)
        combined_pooled_features = avg_pooled_features + max_pooled_features
        out = x * combined_pooled_features
        out = self.relu(self.bn(self.conv3(out)))
        return out

class PSM(nn.Module):
    """ Perceptive Spectrum Manifold (PSM) Module """
    def __init__(self, in_channels, hidden_channels, bottleneck_expansion_ratio=1,
                 expansion_ratio_asymmetric=4, expansion_ratio_asymmetric_2=2):
        super(PSM, self).__init__()
        self.in_channels = in_channels
        # --- Encoder ---
        self.psl1 = PSL(in_channels, hidden_channels)
        bottleneck_channels = hidden_channels * bottleneck_expansion_ratio
        self.psl2 = PSL(hidden_channels * 2, bottleneck_channels)
        # --- Decoder: stage 1 ---
        intermediate_dec1 = bottleneck_channels * expansion_ratio_asymmetric
        self.upsample_e2_psl = nn.ConvTranspose2d(bottleneck_channels * 2,
                                                 bottleneck_channels * 2,
                                                 kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_e2 = nn.BatchNorm2d(bottleneck_channels * 2)
        self.relu_upsample_e2 = nn.ReLU(inplace=True)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 2 + hidden_channels * 2, intermediate_dec1,
                      kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(intermediate_dec1),
            nn.ReLU(inplace=True),

            nn.Conv2d(intermediate_dec1, hidden_channels * 2,
                      kernel_size=(3, 1), padding=(2, 0), dilation=(2, 1), bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        # --- Decoder: stage 2 ---
        intermediate_dec2 = hidden_channels * expansion_ratio_asymmetric_2
        self.upsample_d1_out = nn.ConvTranspose2d(hidden_channels * 2,
                                                  hidden_channels * 2,
                                                  kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_d1 = nn.BatchNorm2d(hidden_channels * 2)
        self.relu_upsample_d1 = nn.ReLU(inplace=True)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + in_channels, intermediate_dec2,
                      kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(intermediate_dec2),
            nn.ReLU(inplace=True),

            nn.Conv2d(intermediate_dec2, in_channels,
                      kernel_size=(3, 1), padding=(2, 0), dilation=(2, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels,96,kernel_size=1,padding=0, bias=False)
        self.bn = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0_skip = x
        # --- Encoder ---
        e1_psl = self.psl1(x0_skip)
        e2_psl = self.psl2(e1_psl)
        # --- Decoder stage 1 ---
        d1_up = self.relu_upsample_e2(self.bn_upsample_e2(self.upsample_e2_psl(e2_psl)))
        d1_cat = torch.cat([d1_up, e1_psl], dim=1)
        d1_out = self.decoder1(d1_cat)
        # --- Decoder stage 2 ---
        d2_up = self.relu_upsample_d1(self.bn_upsample_d1(self.upsample_d1_out(d1_out)))
        d2_cat = torch.cat([d2_up, x0_skip], dim=1)
        x1_fused = self.decoder2(d2_cat)
        # --- Gating ---
        gate = self.sigmoid(x1_fused)
        out = (x1_fused * gate) + x0_skip
        out = self.relu(self.bn(self.conv3(out)))
        return out
        
class myModel(nn.Module):
    def __init__(self, num_classes):
        super(myModel, self).__init__()

        self.conv3x3 = nn.Conv2d(3, 96*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(96*2)
        self.relu = nn.ReLU(inplace=True)

        self.conv3x3_2 = nn.Conv2d(96*2, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_initial_2 = nn.BatchNorm2d(96)

        self.DSSL_module = DSSL()
        self.DSSL_module_2 = DSSL()
        self.DSSL_module_3 = DSSL()

        self.OVR1 = PSM(in_channels=96, hidden_channels=48)
        self.OVR4 = PSM(in_channels=96*2, hidden_channels=48)
        self.OVR2 = PSM(in_channels=96*2, hidden_channels=48)
        self.OVR3 = PSM(in_channels=96*2, hidden_channels=48)

        self.res1_0 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            ASD(channels=96),
        )
        self.ASD = ASD(channels=96*2)

        self.upsample_final1 = nn.ConvTranspose2d(96, 96, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn_upsample_final1 = nn.BatchNorm2d(96)

        self.final_classification_conv = nn.Conv2d(96, num_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Input: (B, 3, 256, 256)
        x = self.relu(self.bn_initial(self.conv3x3(x)))             # (B, 96, 128, 128)
        x = self.relu(self.bn_initial_2(self.conv3x3_2(x)))         # (B, 96, 64, 64)
        x = self.DSSL_module_2(x)

        residual1 = self.res1_0(x)                                  # (B, 96, 64, 64)
        
        x = self.OVR1(x)                                            # (B, 96, 64, 64)
        x = torch.cat([x, residual1], dim=1)                        # (B, 96*2, 64, 64)
        
        residual1 = self.DSSL_module(residual1)
        x = self.OVR2(x)                                            # (B, 96, 64, 64)
        x = torch.cat([x, residual1], dim=1)                        # (B, 96*2, 64, 64)
        
        residual1 = self.DSSL_module_3(residual1)
        x = self.OVR3(x)                                            # (B, 96, 64, 64)
        x = torch.cat([x, residual1], dim=1)                        # (B, 96*2, 64, 64)

        residual1_rd = self.ASD(x)                                  # (B, 96*2, 64, 64)
        out_ovr4 = self.OVR4(x)                                     # (B, 96*2, 64, 64)

        out = out_ovr4*residual1_rd
        out = self.relu(self.bn_upsample_final1(self.upsample_final1(out)))  # (B, 96, 256, 256)
        out = self.final_classification_conv(out)                            # (B, num_classes, 256, 256)

        return out
    

train_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/radarcomm/train/input',
    label_dir='/kaggle/input/radarcomm/train/label',
    transform=train_transform)

val_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/radarcomm/val/input',
    label_dir='/kaggle/input/radarcomm/val/label',
    transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True, num_workers=os.cpu_count())
val_dataloader = DataLoader(val_dataset, batch_size=32, pin_memory=True, shuffle=False, num_workers=os.cpu_count())
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = 4
model = myModel(classes)
def count_parameters(model):  
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(classes) if isinstance(classes, list) else classes 
model = myModel(num_classes)
model = model.to(device)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

num_epochs = 60
epoch_saved = 0
best_mIoU_val = 0.0
best_model_state = None

train_losses = []
val_losses   = []
train_mAccs  = []
val_mAccs    = []
train_mIoUs  = []
val_mIoUs    = []

# --- Train and validationS loop ---
for epoch in range(num_epochs):
    epoch_loss_train, mAcc_train, mIoU_train = train_epoch(model, train_dataloader, criterion, optimizer, device, num_classes)
    epoch_loss_val, mAcc_val, mIoU_val = evaluate(model, val_dataloader, criterion, device, num_classes)
    
    train_losses.append(epoch_loss_train)
    val_losses.append(epoch_loss_val)
    train_mAccs.append(mAcc_train)
    val_mAccs.append(mAcc_val)
    train_mIoUs.append(mIoU_train)
    val_mIoUs.append(mIoU_val)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Learning Rate: {current_lr:.6f}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")

    scheduler.step(epoch_loss_val)
    
    if mIoU_val >= best_mIoU_val:
        epoch_saved = epoch + 1
        best_mIoU_val = mIoU_val
        model_to_save_state_dict = model.module if isinstance(model, nn.DataParallel) else model
        best_model_state = copy.deepcopy(model_to_save_state_dict.state_dict())
        print(f"*** Best epoch at {epoch_saved} with mIoU: {best_mIoU_val:.4f} ***")