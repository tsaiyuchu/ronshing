import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# 1. 讀取 Excel 標籤檔案
label_file = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/datasets/radius.ods"
df = pd.read_excel(label_file, engine='odf')

# 將標籤值減去 1，使範圍變為 [0, 1, 2]
df['Radius'] = df['Radius'] - 1
label_dict = dict(zip(df['No.'], df['Radius']))

# 2. 定義 Dataset 類別
class TumorDataset(Dataset):
    def __init__(self, root_dir, label_dict, transform=None):
        self.root_dir = root_dir
        self.label_dict = label_dict
        self.transform = transform
        self.image_paths, self.targets = self._load_data()

    def _load_data(self):
        image_paths = []
        targets = []

        for patient in sorted(os.listdir(self.root_dir)):
            if patient not in self.label_dict:
                print(f"Skipping {patient} - no label found")
                continue

            tumor_size = self.label_dict[patient]
            for scan_type in ['horizontal', 'vertical']:
                scan_path = os.path.join(self.root_dir, patient, scan_type)
                for img_file in sorted(glob.glob(os.path.join(scan_path, '*.png'))):
                    image_paths.append(img_file)
                    targets.append(tumor_size)

        return image_paths, targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, target

# 3. 定義資料增強技術與資料轉換
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 初始化 Dataset
root_dir = '/home/yuchu/Downloads/ronshing_original/Ronshing-001/Ronshing/Swin-Unet/raw_image/trainval/'
dataset = TumorDataset(root_dir=root_dir, label_dict=label_dict, transform=transform)

# 5. 定義 K 折交叉驗證
k_folds = 5
num_epochs = 100
batch_size = 16
learning_rate = 1e-4

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K 折交叉驗證
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    # 分割訓練集和驗證集
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 使用 ResNet50 並調整輸出層
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 3)
    )

    # 將模型移動到裝置上
    model = model.to(device)

    # 損失函數和優化器設置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 訓練與驗證
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # 驗證階段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(epoch + len(train_loader))

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"radius_kkfold{fold + 1}.pth")
            print(f"模型已更新並儲存為 radius_kkfold{fold + 1}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

    # 混淆矩陣與準確率計算
    def evaluate_with_confusion(model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device).long()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        cm = confusion_matrix(all_targets, all_preds)
        accuracy = accuracy_score(all_targets, all_preds)

        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

    # 呼叫混淆矩陣驗證函數
    evaluate_with_confusion(model, val_loader)
