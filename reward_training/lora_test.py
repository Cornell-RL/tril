import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from copy import deepcopy
import pdb

from tqdm import tqdm

from datetime import timedelta
from datasets import load_dataset

TOKENIZER_CFG = {
    "model_name": "EleutherAI/gpt-j-6B",
    "pad_token_as_eos_token": True,
    "padding_side": "right",
    "truncation_side": "right",
}

DATAPOOL_CFG = {
    "id": "tldr_preference"
}

cfg = {
    "lr": 6e-5,
    "weight_decay": 0,
    "n_epochs": 1,
    "eval_interval": 1500, # Roughly 4 times per epoch
    "gradient_accumulation": 2,
    "lora_r": 32
}

# Model (QLora Training)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL", # Automatically saves scores layer
)
model = AutoModelForCausalLM.from_pretrained("gpt2-xl",
                                               quantization_config=nf4_config)
#model.config.pad_token_id = tokenizer.pad_token_id # pad token to special token
model = get_peft_model(model, peft_config=peft_config)
import pdb; pdb.set_trace()




