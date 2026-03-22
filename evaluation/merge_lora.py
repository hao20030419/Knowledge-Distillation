import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    # 你的微調 checkpoint 路徑，例如 checkpoint-qwen-ft/checkpoint-100
    parser.add_argument("--adapter_dir", type=str, required=True, help="Path to the LoRA adapter")
    # 輸出合併後模型的路徑
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the merged model")
    args = parser.parse_args()

    print(f"Loading adapter from {args.adapter_dir}...")
    
    # AutoPeftModel 會自動讀取 base_model_name_or_path 並載入基礎模型與 LoRA
    # device_map="cpu" 是為了避免顯存不足（合併過程建議在 CPU RAM 做，除非你有 24GB+ VRAM）
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cpu"  
    )

    print("Merging model weights...")
    # 這是關鍵步驟：將 LoRA 權重合併進 Base Model
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    
    # 記得也要存 tokenizer，因為之後載入我們要改用這個新路徑
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Done! You can now use the merged model for faster inference.")

if __name__ == "__main__":
    main()