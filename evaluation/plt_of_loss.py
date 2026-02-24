import json
import matplotlib.pyplot as plt
import numpy as np

# 1. 設定檔案路徑
file_path = r"D:\Hao\Knowledge-Distillation\checkpoint-qwen7B-ft4000\checkpoint-1000\trainer_state.json"

def plot_optimized_loss(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    history = data.get("log_history", [])
    steps = [entry["step"] for entry in history if "loss" in entry]
    losses = [entry["loss"] for entry in history if "loss" in entry]
    epochs = [entry["epoch"] for entry in history if "loss" in entry]

    if not steps: return

    plt.figure(figsize=(15, 8))
    
    # 畫出原始數據（淡藍色，不帶點）
    plt.plot(steps, losses, color='#1f77b4', alpha=0.3, label='Raw Loss')
    
    # 畫出平滑曲線 (移動平均)
    window_size = 5
    if len(losses) > window_size:
        smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(steps[window_size-1:], smooth_losses, color='blue', linewidth=2, label=f'Smooth Loss (MA-{window_size})')

    # --- 關鍵優化：每隔 N 個點才標註一次 Loss ---
    annotate_every = 5 # 每 5 個點標一個，可以根據視覺調整
    for i in range(0, len(steps), annotate_every):
        plt.annotate(f"{losses[i]:.3f}", 
                     (steps[i], losses[i]),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8, color='red', fontweight='bold')
    
    # 特別標註最後一個點
    plt.annotate(f"End: {losses[-1]:.3f}", 
                 (steps[-1], losses[-1]),
                 textcoords="offset points", xytext=(15, -5),
                 ha='left', fontsize=9, color='darkred', fontweight='extra bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    # 優化 X 軸刻度（只顯示部分 Step，避免擠爆）
    plt.xticks(steps[::annotate_every], rotation=45, fontsize=8)
    
    plt.title('Training Loss Analysis', fontsize=16)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    #plt.savefig("loss_optimized.png", dpi=300)
    plt.show()

plot_optimized_loss(file_path)