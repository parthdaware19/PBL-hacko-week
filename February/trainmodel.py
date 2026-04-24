

import os

# Install required packages quietly
print("Installing necessary packages. This will take a few minutes...")
os.system("pip install -q -U transformers peft accelerate trl datasets bitsandbytes huggingface_hub")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import notebook_login

print("Please log in to Hugging Face when prompted (requires a WRITE token)")
notebook_login()

HF_USERNAME = input("Enter your Hugging Face username (e.g., john-doe): ").strip()
OUTPUT_MODEL_NAME = f"{HF_USERNAME}/healthcare-chatbot-tinyllama"

# ================================
# 1. Setup Model and Tokenizer
# ================================
# We use TinyLlama-1.1B because it is fast to train and runs extremely well on a T4 GPU.
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# QLoRA configuration for 4-bit quantization (drastically reduces VRAM usage)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading tokenizer and model ({model_id}) in 4-bit mode...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure padding token is set for batched training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# ================================
# 2. Load Medical Dataset
# ================================
print("Loading medical dataset (ChatDoctor-HealthCareMagic)...")
# We load a small 5000-sample subset to keep training time under ~30-60 mins for a quick prototype.
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train[:5000]")

def format_instruction(sample):
    # Formatting prompt for the Llama-2 / ChatML style 
    return f"<|system|>\nYou are a helpful medical assistant. However, you are not a replacement for a doctor. Always recommend seeing a professional.\n<|user|>\n{sample['instruction']}\n<|assistant|>\n{sample['output']}"

def prepare_dataset(example):
    example["text"] = format_instruction(example)
    return example

dataset = dataset.map(prepare_dataset)

# ================================
# 3. LoRA Configuration
# ================================
print("Applying LoRA adapters...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# ================================
# 4. Training Arguments
# ================================
training_args = TrainingArguments(
    output_dir="./medical-chatbot-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=200,    # Max steps to keep it brief. Increase this up to 1000 for better results.
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

# ================================
# 5. Trainer Initialization
# ================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# ================================
# 6. Start Training
# ================================
print("Starting Training! Grab a coffee ☕...")
trainer.train()

# ================================
# 7. Push to Hugging Face
# ================================
print(f"Training Complete! Pushing adapter weights to {OUTPUT_MODEL_NAME}...")
trainer.model.push_to_hub(OUTPUT_MODEL_NAME)
tokenizer.push_to_hub(OUTPUT_MODEL_NAME)

print(f"✅ Success! Your model is now live at: https://huggingface.co/{OUTPUT_MODEL_NAME}")
print("You can plug this repository name into the web app's app.py.")