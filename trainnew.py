import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score  # 新增這行

# 1. 讀取 Excel 標籤檔案
label_file = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/datasets/Near.ods"
df = pd.read_excel(label_file, engine='odf')

# 建立標籤字典
label_dict = dict(zip(df['No.'], df['Near']))

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
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target

# 3. 定義資料增強技術與資料轉換
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 初始化 Dataset 和 DataLoader
train_dir = '/home/yuchu/ronshing/Ronshing-001/Ronshing/Swin-Unet/datattv/train/'
val_dir = '/home/yuchu/ronshing/Ronshing-001/Ronshing/Swin-Unet/datattv/val/'

train_dataset = TumorDataset(root_dir=train_dir, label_dict=label_dict, transform=transform)
val_dataset = TumorDataset(root_dir=val_dir, label_dict=label_dict, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 5. 使用 ResNet18 並調整輸出層
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# 6. 訓練與驗證
num_epochs = 300
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images).squeeze(1)
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
            images, targets = images.to(device), targets.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("模型已更新並儲存為 best_model.pth")

# 7. 混淆矩陣與準確率計算
def evaluate_with_confusion(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images).squeeze(1).round()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# 呼叫混淆矩陣驗證函數
evaluate_with_confusion(model, val_loader)
