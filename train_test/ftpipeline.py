from datasets import load_dataset
from random import randrange

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

from trl import SFTTrainer

# The model that you want to train from the Hugging Face hub
model_id = "NousResearch/Llama-2-7b-hf"
# The instruction dataset to use
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
#dataset_name = "HuggingFaceH4/CodeAlpaca_20K"
# Dataset split
dataset_split= "train"
# Fine-tuned model name
new_model = "llama-2-7b-int4-python-code-20k"
# Huggingface repository
hf_model_repo="edumunozsala/"+new_model
# Load the entire model on the GPU 0
device_map = {"": 0}

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_double_nested_quant = False

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = new_model
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True
# Batch size per GPU for training
per_device_train_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1 # 2
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 #1e-5
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine" #"constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False
# Save checkpoint every X updates steps
save_steps = 0
# Log every X updates steps
logging_steps = 25
# Disable tqdm
disable_tqdm= True

################################################################################
# SFTTrainer parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 2048 #None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True #False


from huggingface_hub import notebook_login
# Log in to HF Hub
notebook_login()