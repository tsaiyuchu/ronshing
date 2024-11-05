import os
import glob
import torch
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn  # 確保 nn 被匯入

# 1. 測試資料路徑與模型權重檔案
test_data_dir = "/home/yuchu/Downloads/ronshing_original/Ronshing-001/Ronshing/Swin-Unet/raw_image/test/"
model_path = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/radius_kfold2.pth"
label_file = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/datasets/radius.ods"

# 2. 資料轉換（與訓練時一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 確保圖片大小與訓練一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 初始化模型並載入訓練好的權重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用與訓練時一致的 ResNet50 模型
model = models.resnet50()  # 使用預訓練權重進行初始化
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 3)  # 更新輸出層的類別數量為 3
)

# 載入模型權重
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 以允許某些層可能不匹配
model.to(device)
model.eval()

# 4. 建立標籤字典
df = pd.read_excel(label_file, engine='odf')
label_dict = dict(zip(df['No.'], df['Radius']))

# 5. 測試函數
def test_model(model, data_dir, label_dict):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patient in sorted(os.listdir(data_dir)):
            if patient not in label_dict:
                print(f"Skipping {patient} - no label found")
                continue

            true_radius = label_dict[patient]  # 真實值加 1 以匹配範圍 [1, 2, 3]
            patient_dir = os.path.join(data_dir, patient)

            predicted_sum = 0
            image_count = 0

            for scan_type in ['horizontal', 'vertical']:
                scan_path = os.path.join(patient_dir, scan_type)

                if not os.path.exists(scan_path):
                    continue

                for img_path in sorted(glob.glob(os.path.join(scan_path, "*.png"))):
                    # 處理影像
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image).unsqueeze(0).to(device)

                    # 預測並累加
                    outputs = model(image)
                    _, predicted = torch.max(outputs, 1)
                    predicted_sum += predicted.item()
                    image_count += 1

            # 確保有處理到至少一張影像
            if image_count > 0:
                # 計算平均預測值並四捨五入
                avg_predicted_radius = round(predicted_sum / image_count) + 1  # 預測值加 1 以匹配範圍 [1, 2, 3]
            else:
                avg_predicted_radius = 1  # 如果沒有影像，預設為 1

            # 儲存預測和真實值
            all_preds.append(avg_predicted_radius)
            all_targets.append(true_radius)

            print(f"Patient: {patient}, True Radius: {true_radius}, "
                  f"Predicted Average Radius: {avg_predicted_radius}")

    # 計算混淆矩陣與準確率
    cm = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return all_preds, all_targets, accuracy

# 6. 執行測試並儲存結果
all_preds, all_targets, accuracy = test_model(model, test_data_dir, label_dict)

# 7. 將結果寫入 CSV 檔案
output_file = "test_results_radius.csv"
df_results = pd.DataFrame({
    "Patient": sorted(os.listdir(test_data_dir)),
    "Radius": all_targets,
    "Predicted_Average_Radius": all_preds
})
df_results.to_csv(output_file, index=False)
print(f"測試結果已儲存至 {output_file}")
