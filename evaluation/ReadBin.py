import torch
from transformers import TrainingArguments

# 設定你的檔案路徑
file_path = r"D:\Hao\Knowledge-Distillation\checkpoint-qwen7B-ft4000\training_args.bin"

try:
    # 載入資料
    # weights_only=False 是因為這是一個物件實例而非單純的張量數值
    args = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
    
    print("--- 成功讀取訓練參數 ---")
    
    # 將參數轉換為字典格式以便查看
    args_dict = args.to_dict()
    
    for key, value in args_dict.items():
        print(f"{key}: {value}")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_path}")
except Exception as e:
    print(f"讀取時發生錯誤：{e}")