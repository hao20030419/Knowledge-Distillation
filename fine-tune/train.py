import argparse
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_utils import unwrap_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
try:
    from bitsandbytes import __version__ as bnb_version
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


class DataCollatorForCausalLM:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in labels], batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_example(example, tokenizer, max_length=1024):
    # Expect `messages` list: messages[0] user, messages[1] assistant
    msgs = example.get("messages") or []
    if isinstance(msgs, list) and len(msgs) >= 2:
        prompt = msgs[0].get("content", "")
        response = msgs[1].get("content", "")
    else:
        # fallback to fields
        prompt = example.get("question", "") or example.get("prompt", "")
        response = example.get("answer", "") or example.get("completion", "")

    # For causal LM: concatenate and mask prompt tokens in labels
    sep = tokenizer.eos_token or "\n"
    full = prompt + sep + response + (tokenizer.eos_token or "")
    full_ids = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=True).input_ids
    prompt_ids = tokenizer(prompt + sep, truncation=True, max_length=max_length, add_special_tokens=True).input_ids

    labels = full_ids.copy()
    # mask prompt part
    for i in range(len(prompt_ids)):
        labels[i] = -100
    return {"input_ids": full_ids, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="../GeminiAgent/results/clean_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./qwen-finetuned")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization via bitsandbytes")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization via bitsandbytes (bnb) - requires bitsandbytes and transformers support")
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.dataset_path, split="train", keep_in_memory=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|endoftext|>"})

    # Map dataset
    def map_fn(example):
        return build_example(example, tokenizer, max_length=args.max_length)

    ds = ds.map(map_fn, remove_columns=ds.column_names)

    # Load model with optional bitsandbytes quantization
    print("Loading model (this may take a while)...")
    load_kwargs = {}
    if args.load_in_4bit or args.load_in_8bit:
        if not BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes is required for 8-bit/4-bit loading. Install bitsandbytes and a compatible transformers version.")

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif args.load_in_8bit:
        # 8-bit via load_in_8bit and device_map auto
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
        load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **load_kwargs,
        low_cpu_mem_usage=True,
    )

    # Prepare for LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # If using kbit quantization, prepare model accordingly
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # prepare_model_for_kbit_training may not be necessary for full precision models
        pass
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorForCausalLM(tokenizer, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()
    model.push_to_hub = False
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
