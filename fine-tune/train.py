import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Force GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    from bitsandbytes import __version__ as bnb_version
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model identifier")
    parser.add_argument("--dataset_path", type=str, default="../GeminiAgent/results/clean_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./qwen-finetuned")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    # 1. Standardize Dataset Format
    # Ensure every example has "messages" format for Qwen
    def standardize_data(example):
        msgs = example.get("messages") or []
        if isinstance(msgs, list) and len(msgs) >= 2:
            return {"messages": msgs}
        else:
            # Fallback for "question"/"answer" or "prompt"/"completion"
            prompt = example.get("question", "") or example.get("prompt", "")
            response = example.get("answer", "") or example.get("completion", "")
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
    
    ds = ds.map(standardize_data)

    # 2. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Model (with Quantization support)
    print("Loading model...")
    bnb_config = None
    if args.load_in_4bit:
        if not BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes required for 4-bit loading")
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Gradient Checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Optimized for Qwen
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        fp16=True, # Qwen supports fp16/bf16
        warmup_ratio=0.05,
        optim="paged_adamw_32bit" if args.load_in_4bit else "adamw_torch",
        gradient_checkpointing=True,
    )

    # 6. Formatting Function for SFTTrainer
    # This prepares the text for training
    def formatting_prompts_func(examples):
        output_texts = []
        for messages in examples['messages']:
            # Apply standard Qwen Chat Template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # 7. Data Collator (Masking Prompt)
    # Automatically masks the user instructions so loss is only calculated on assistant response
    # Qwen-Instruct typically uses "<|im_start|>assistant\n"
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # 8. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_prompts_func,
        data_collator=collator, 
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
