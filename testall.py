import os
import glob
import torch
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

# 1. 測試資料路徑與模型權重檔案
test_data_dir = "/home/yuchu/ronshing/Ronshing-001/Ronshing/Swin-Unet/datattv/test/"
models_paths = {
    'Radius': "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/radius_3.pth",
    'Exo/endo': "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/exo_3.pth",
    'Near': "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/near_1.pth",
    'A/P/X': "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/apx_2.pth",
    'Location': "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/code/main/location_2.pth"
}

# 2. 資料轉換（與訓練時一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 確保圖片大小與訓練一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 初始化模型加載函數
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.to(device)
    model.eval()
    return model

# 4. 修改建立標籤字典的函數，從合併的單一 .ods 文件提取標籤，處理 A/P/X 的轉換
def load_labels(merged_label_file):
    df = pd.read_excel(merged_label_file, engine='odf')

    # 提取所有標籤字典，對應的欄位名稱必須與表頭一致
    label_dicts = {
        'Radius': dict(zip(df['No.'], df['Radius'])),
        'Exo/endo': dict(zip(df['No.'], df['Exo/endo'])),
        'Near': dict(zip(df['No.'], df['Near'])),
        'A/P/X': dict(zip(df['No.'], df['A/P/X'].map({'a': 1, 'p': 2, 'x': 3}))),  # 將 a, p, x 轉換為 1, 2, 3
        'Location': dict(zip(df['No.'], df['Location']))
    }
    
    return label_dicts

# 5. 測試函數
def test_model(model, data_dir, label_dict):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patient in sorted(os.listdir(data_dir)):
            if patient not in label_dict:
                print(f"Skipping {patient} - no label found")
                continue

            true_label = label_dict[patient]  # 真實值
            patient_dir = os.path.join(data_dir, patient)

            predicted_sum = 0
            image_count = 0

            for scan_type in ['horizontal', 'vertical']:
                scan_path = os.path.join(patient_dir, scan_type)
                for img_path in sorted(glob.glob(os.path.join(scan_path, "*.png"))):
                    # 處理影像
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image).unsqueeze(0).to(device)

                    # 預測並累加
                    predicted_value = model(image).squeeze(1).item()
                    predicted_sum += predicted_value
                    image_count += 1

            # 計算平均預測值並四捨五入
            avg_predicted_value = round(predicted_sum / image_count)

            # 儲存預測和真實值
            all_preds.append(avg_predicted_value)
            all_targets.append(true_label)

    return all_preds, all_targets

# 6. 執行測試
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入單一合併標籤文件
merged_label_file = "/home/yuchu/ronshing/Ronshing-001/Ronshing/renal_scoring/datasets/all_label.ods"  # 修改為你的合併 .ods 文件的路徑
label_dicts = load_labels(merged_label_file)

final_results = {}
accuracies = {}

# 對每個模型進行測試
for label_name, model_path in models_paths.items():
    print(f"Testing for {label_name}...")
    model = load_model(model_path)
    label_dict = label_dicts[label_name]
    
    preds, targets = test_model(model, test_data_dir, label_dict)
    final_results[label_name] = (preds, targets)

# 7. 先輸出每個病患的標籤值
def convert_apx_label(value):
    """ Convert numerical A/P/X back to 'a', 'p', 'x' """
    mapping = {1: 'a', 2: 'p', 3: 'x'}
    return mapping.get(value, 'unknown')

patients = sorted(os.listdir(test_data_dir))
for i, patient in enumerate(patients):
    true_labels = [final_results[label][1][i] for label in models_paths.keys()]
    predicted_labels = [final_results[label][0][i] for label in models_paths.keys()]

    # Convert A/P/X predictions and true values back to 'a', 'p', 'x'
    true_labels[3] = convert_apx_label(true_labels[3])  # A/P/X is the 4th label (index 3)
    predicted_labels[3] = convert_apx_label(predicted_labels[3])

    print(f"Patient: {patient}, True Labels: {true_labels}, Predicted Labels: {predicted_labels}")

# 8. 分別計算和輸出每個標籤的準確率
for label_name, result in final_results.items():
    if label_name == 'A/P/X':
        # Convert numeric to 'a', 'p', 'x' for accuracy calculation
        true_apx = [convert_apx_label(x) for x in result[1]]
        predicted_apx = [convert_apx_label(x) for x in result[0]]
        accuracy = accuracy_score(true_apx, predicted_apx)
    else:
        accuracy = accuracy_score(result[1], result[0])
    
    accuracies[label_name] = accuracy
    print(f"{label_name} Accuracy: {accuracy * 100:.2f}%")

# 9. 計算所有標籤都正確的準確率
correct_predictions = 0
total_patients = len(patients)

for i in range(total_patients):
    all_correct = True
    for label_name in models_paths.keys():
        if label_name == 'A/P/X':
            if convert_apx_label(final_results['A/P/X'][0][i]) != convert_apx_label(final_results['A/P/X'][1][i]):
                all_correct = False
                break
        elif final_results[label_name][0][i] != final_results[label_name][1][i]:
            all_correct = False
            break
    if all_correct:
        correct_predictions += 1

overall_accuracy = correct_predictions / total_patients
print(f"Overall Accuracy (all labels correct): {overall_accuracy * 100:.2f}%")

# 10. 儲存結果至 CSV 檔案
results_dict = {"Patient": patients}
for label in models_paths.keys():
    results_dict[f"True_{label}"] = final_results[label][1]
    results_dict[f"Predicted_{label}"] = final_results[label][0]

output_file = "test_results_combined.csv"
df_results = pd.DataFrame(results_dict)
df_results.to_csv(output_file, index=False)
print(f"測試結果已儲存至 {output_file}")
