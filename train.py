import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model  # Import peft
import argparse
import deepspeed
import json

# --- Configuration ---
# Use DeepSpeed ZeRO stage 3 for model parallelism.
deepspeed_config = {
    "train_batch_size": 1,  # Adjust based on your GPU memory
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-4,
            "warmup_num_steps": 100,
        }
    },

    "zero_optimization": {
        "stage": 3, # VERY Important.
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "load_from_fp32_weights": False, #Since loading a 4-bit model
         "offload_optimizer": { #For offloading optimizer state
                "device": "cpu",
                "pin_memory": True
            },
        "offload_param": { #For offloading parameters
                "device": "cpu",
                "pin_memory": True
            },
    },
      "fp16": {
        "enabled": "auto", #Let DeepSpeed decide whether to use FP16
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "bf16": { #If you can use bf16 then use this.
        "enabled": "auto" # Let DeepSpeed decide.
    },
    "wall_clock_breakdown": False
}

# --- Dataset Class (Simplified for Demonstration) ---
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
          for item in data:
            self.examples.append(item["text"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # For causal language modeling, the labels are the input itself, shifted by one.
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),  # Labels are the same as input_ids
        }

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Pretrained model name or path.")
    parser.add_argument("--data_file", type=str, default="data/preprocessed/preprocessed.json", help="Path to the preprocessed data file.")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for trained model.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.") #For DDP
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to the DeepSpeed configuration file.") #DeepSpeed Config
    args = parser.parse_args()


    # --- Initialization (Distributed) ---
    # Initialize the process group
    deepspeed.init_distributed()

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: # Add pad token if not already present
        tokenizer.pad_token = tokenizer.eos_token
    # --- Dataset ---
    train_dataset = TextDataset(args.data_file, tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=1)  # Keep batch size 1 with DDP and model parallelism

    # --- Model ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))}, #VERY IMPORTANT, to use the current device.
    )
    # --- LoRA Configuration ---
    peft_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,  # Alpha
        lora_dropout=0.05,  # Dropout
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] #For mistral
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1, # Keep this to 1
        gradient_accumulation_steps=1,  # Accumulate gradients over multiple steps
        learning_rate=2e-4,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,      # Log every 10 steps
        save_strategy="steps", # Save the model and optimizer state
        save_steps=500,        # Save every 500 steps
        save_total_limit=2,    # Keep only the last 2 checkpoints
        report_to="tensorboard",
        deepspeed=deepspeed_config #DeepSpeed configuration
    )


    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {  # Simple data collator
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data]),
        },
    )

    # --- Training ---
    trainer.train()
    # --- Save Model ---
    peft_model_id = f"{args.output_dir}/final_checkpoint"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    #trainer.save_model() # Save the trained model and tokenizer

if __name__ == "__main__":
    main()
