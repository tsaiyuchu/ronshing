import os
import glob
import torch
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np

# 1. 測試資料路徑與模型權重檔案
test_data_dir = "/home/yuchu/ronshing/Ronshing-001/Ronshing/Swin-Unet/datattv/test/"
model_path = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/all_3.pth"
label_file = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/datasets/all_label.ods"

# 2. 資料轉換（與訓練時一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 確保圖片大小與訓練一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 初始化模型並載入訓練好的權重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用 ResNet18（與訓練時一致）
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 15)  # 5 個標籤，每個標籤 one-hot 長度為 3，共計 15
)

# 載入模型權重
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.to(device)
model.eval()

# 4. 建立標籤字典
def apx_mapping(value):
    if value == "a":
        return 1
    elif value == "p":
        return 2
    elif value == "x":
        return 3
    else:
        return 1  # 預設值

df = pd.read_excel(label_file, engine='odf')
df['A/P/X'] = df['A/P/X'].map(apx_mapping)
df[['Radius', 'Exo/endo', 'Near', 'A/P/X', 'Location']] = df[['Radius', 'Exo/endo', 'Near', 'A/P/X', 'Location']].astype(int)

label_dict = {row['No.']: row[['Radius', 'Exo/endo', 'Near', 'A/P/X', 'Location']].values for _, row in df.iterrows()}

# 5. 測試函數
def test_model(model, data_dir, label_dict):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patient in sorted(os.listdir(data_dir)):
            if patient not in label_dict:
                print(f"Skipping {patient} - no label found")
                continue

            true_labels = label_dict[patient]  # 真實標籤
            patient_dir = os.path.join(data_dir, patient)

            predicted_sum = torch.zeros(5, dtype=torch.long).to(device)  # 預測的累積結果，5個標籤
            image_count = 0

            for scan_type in ['horizontal', 'vertical']:
                scan_path = os.path.join(patient_dir, scan_type)

                for img_path in sorted(glob.glob(os.path.join(scan_path, "*.png"))):
                    # 處理影像
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image).unsqueeze(0).to(device)

                    # 模型預測
                    predicted_logits = model(image).view(5, 3)  # 每個標籤3個類別 (5個標籤, 3類)
                    predicted_labels = torch.argmax(predicted_logits, dim=1) + 1  # 找出最大機率的類別，並將其轉換為1, 2, 3

                    predicted_sum += predicted_labels
                    image_count += 1

            # 計算每個標籤的平均預測值
            avg_predicted_labels = torch.round(predicted_sum / image_count).int()

            # 儲存預測和真實值
            all_preds.append(avg_predicted_labels.cpu().numpy())
            all_targets.append(true_labels)

            print(f"Patient: {patient}, True Labels: {true_labels}, "
                  f"Predicted Labels: {avg_predicted_labels.tolist()}")

    # 6. 將 all_targets 和 all_preds 轉為 numpy 陣列並確保它們為數值型別
    all_targets = np.array(all_targets, dtype=int)
    all_preds = np.array(all_preds, dtype=int)

    # 計算每個標籤的準確率
    label_names = ['Radius', 'Exo/endo', 'Near', 'A/P/X', 'Location']
    label_accuracies = {}

    for i, label_name in enumerate(label_names):
        accuracy = accuracy_score(all_targets[:, i], all_preds[:, i])
        label_accuracies[label_name] = accuracy
        print(f"{label_name} Accuracy: {accuracy * 100:.2f}%")

    # 計算所有標籤都正確的準確率
    overall_accuracy = np.mean(np.all(all_targets == all_preds, axis=1))
    print(f"Overall Accuracy (all labels correct): {overall_accuracy * 100:.2f}%")

    return label_accuracies, overall_accuracy


# 7. 執行測試並儲存結果
label_accuracies, overall_accuracy = test_model(model, test_data_dir, label_dict)

# 8. 將結果寫入 CSV 檔案
output_file = "test_results_multilabel.csv"
df_results = pd.DataFrame({
    "Patient": sorted(os.listdir(test_data_dir)),
    "Radius_Accuracy": label_accuracies['Radius'],
    "Exo_endo_Accuracy": label_accuracies['Exo/endo'],
    "Near_Accuracy": label_accuracies['Near'],
    "A_P_X_Accuracy": label_accuracies['A/P/X'],
    "Location_Accuracy": label_accuracies['Location'],
    "Overall_Accuracy": overall_accuracy
})
df_results.to_csv(output_file, index=False)
print(f"測試結果已儲存至 {output_file}")
