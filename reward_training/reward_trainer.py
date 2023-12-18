import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
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

def plot_calibration(model_name, dataset_name, delta_scores):
    space = np.linspace(0, 4, 32)
    perfect_calibration = 1 / (1 + np.exp(-space))

    epsilon = 1 / 4
    probs = []
    for center in space:
        ixs = (center - epsilon < abs(delta_scores)) & (abs(delta_scores) < center + epsilon)
        if not ixs.any():
            prob = 0.5
        else:
            prob = np.mean(delta_scores[ixs] > 0)

        probs.append(prob)

    import matplotlib
    from matplotlib import pyplot

    textcolor = "#333"
    matplotlib.style.use("ggplot")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 15,
        "text.color": textcolor,
        "axes.labelcolor": textcolor,
        "axes.labelpad": 12,
        "xtick.color": textcolor,
        "ytick.color": textcolor,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.titlesize": 14,
        "figure.figsize": (12, 8),
    })
    pyplot.plot(space, perfect_calibration, label="perfect calibration", c="grey")
    pyplot.plot(space, probs, label=model_name)

    ax = pyplot.gca()
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)
    ax.set_facecolor("#fff")
    ax.set_title(f"Calibration on {dataset_name}", size=26, y=1.02, fontdict={"fontweight": "normal"})
    ax.set_xlabel("Score difference", size=26)
    ax.set_ylabel("Accuracy", size=26)
    pyplot.legend(loc="best", fontsize=20, title_fontproperties={"weight": "normal", "style": "normal"}, fancybox=False, frameon=False)
    pyplot.tight_layout()

    os.makedirs("calibrations", exist_ok=True)
    image_path = os.path.join("calibrations", f"{model_name}@{dataset_name}.png".replace("/", "_"))
    pyplot.savefig(image_path, dpi=64)

def get_ref(
    model_id,
    tokenizer_id="EleutherAI/gpt-j-6b",
    adapter_id="16_gptj_process",
):
    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[kwargs]
    )
    max_seq_length=550
    eval_batch_size=32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # ADD a special pad token. EOS TOKEN used for something specific
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left" # focus on the generations

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
        #modules_to_save=["score"],
        task_type="SEQ_CLS", # Automatically saves scores layer
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               quantization_config=nf4_config,
                                                               num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id # pad token to special token
    model.resize_token_embeddings(len(tokenizer)) # resize embeddings for pad token
    #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config=peft_config)
    model.load_adapter(adapter_id, 'default')

    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    #dataset = dataset['train']
    #dataset = dataset['test']
    dataset = dataset['valid']

    def reformat(sample):
        prompt = sample['prompt'].split("TL;DR:")[0].strip()
        prompt = prompt.replace("\nTITLE:", "\n\nTITLE:") \
                       .replace("\nPOST:", "\n\nPOST:")
        #reference = "TL;DR: " + sample['label'] + tokenizer.eos_token
        reference = "TL;DR: " + sample['label'].strip()
        #text = "\n\n".join([prompt, reference])
        return {"prompt": prompt, "reference": reference}

    def tokenize(prompt, reference, tokenizer):
        prompt = tokenizer.decode(
            tokenizer(
                prompt,
                truncation=True,
                max_length=500,
                add_special_tokens=False,
            )['input_ids'],
            skip_special_tokens=True,
        )
        text = "\n\n".join([prompt, reference])
        return {
            "input_ids": tokenizer(
                text + tokenizer.eos_token, truncation=True, max_length=max_seq_length
            ).input_ids
        }

    def collate_fn(batch):
        input_ids = [x["input_ids"] for x in batch]
        return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

    dataset = dataset.map(reformat, desc="Reformatting")
    tokenized = dataset.map(
        tokenize,
        input_columns=["prompt", "reference"],
        fn_kwargs=dict(tokenizer=tokenizer),
        desc="Tokenizing"
    )
    dataloader = torch.utils.data.DataLoader(
        tokenized, shuffle=False, batch_size=eval_batch_size, collate_fn=collate_fn
    )
    optimizer = Adam(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-08
    )
    # Accelerate Prepare
    model, dataloader, _ = accelerator.prepare(model, dataloader, optimizer)
    all_scores = []
    for batch in tqdm(dataloader, desc=f"Evaluating Reference", disable=not accelerator.is_main_process):
        with torch.no_grad():
            scores = model(**batch)[0]
            scores = accelerator.gather_for_metrics(scores)
            all_scores.extend(scores.tolist())

    if accelerator.is_main_process:
        scores = np.hstack(all_scores)
        mean, std, low, high = scores.mean(), scores.std(), scores.min(), scores.max()
        print(f"Avg: {mean} | Std: {std} | Min: {low} | Max: {high}")


def test_eval(model_id):
    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        #log_with="wandb",
        kwargs_handlers=[kwargs]
    )
    #model_id = "/home/jdc396/tril/outputs/pretrain_gptxl_reformat/2023-11-21_15-50-01/model_2000"
    #tokenizer_id="gpt2-xl"
    tokenizer_id="EleutherAI/gpt-j-6b"
    max_seq_length=550
    eval_batch_size=16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # ADD a special pad token. EOS TOKEN used for something specific
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #tokenizer.pad_token_id = tokenizer.eos_token_id #BUG
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left" # focus on the generations

    def tokenize(prompt, selected, rejected, tokenizer):
        # NOTE: for TL;DR, TL;DR: prefix is in each selected and rejected
        #prompt = tokenizer.decode(tokenizer(prompt, truncation=True, max_length 
        #processed_prompt = [p.split("TL;DR:")[0] for p in prompt]
        prompt = tokenizer.decode(
            tokenizer(
                prompt,
                truncation=True,
                max_length=500,
                #- 5,  # to make sure "TL;DR" dont get truncated
                #- 7,  # to make sure "TL;DR" dont get truncated
                add_special_tokens=False,
            )["input_ids"],
            skip_special_tokens=True,
        )
        return {
            "selected_input_ids": tokenizer(
                prompt + "\n\n" + selected + tokenizer.eos_token, truncation=True, max_length=max_seq_length
            ).input_ids,
            "rejected_input_ids": tokenizer(
                prompt + "\n\n" + rejected + tokenizer.eos_token, truncation=True, max_length=max_seq_length
            ).input_ids,
        }

    def collate_fn(batch):
        #input_ids = sum([[x["selected_input_ids"], x["rejected_input_ids"]] for x in batch], [])
        input_ids = [x["selected_input_ids"] for x in batch] + [x["rejected_input_ids"] for x in batch]
        return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

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
        #modules_to_save=["score"],
        task_type="SEQ_CLS", # Automatically saves scores layer
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               quantization_config=nf4_config,
                                                               num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id # pad token to special token
    model.resize_token_embeddings(len(tokenizer)) # resize embeddings for pad token
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config=peft_config)
    #model.load_adapter("test_rm", 'default')
    #model.load_adapter("test_79", 'default')
    model.load_adapter("16_gptj_process_final", 'default')


    # Dataset
    def reformat(sample):
        prompt = sample['prompt'].strip()
        prompt = prompt.replace("\nTITLE:", "\n\nTITLE:") \
                       .replace("\nPOST:", "\n\nPOST:")

        selected = sample["selected"].strip()
        rejected = sample["rejected"].strip()
        return {"prompt": prompt, "selected": selected, "rejected": rejected}

    dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="test")
    if "chosen" in dataset.column_names:
        dataset = dataset.rename_column("chosen", "selected")
    dataset = dataset.map(reformat, desc="Reformatting")
    tokenized = dataset.map(
        tokenize,
        input_columns=["prompt", "selected", "rejected"],
        fn_kwargs=dict(tokenizer=tokenizer),
        desc="Tokenizing"
    )
    test_dataloader = torch.utils.data.DataLoader(
        tokenized, shuffle=False, batch_size=eval_batch_size, collate_fn=collate_fn
    )

    #optimizer = AdamW(
    #    model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-08, weight_decay=cfg["weight_decay"]
    #)
    optimizer = Adam(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-08
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3000)

    # Accelerate Prepare
    model, optimizer, scheduler, test_dataloader = accelerator.prepare(
        model, optimizer, scheduler, test_dataloader
    )

    model.eval()
    all_delta_scores = []
    for eval_batch in tqdm(test_dataloader, desc=f"Evaluating", disable=not accelerator.is_main_process, leave=False):
        with torch.no_grad():
            scores = model(**eval_batch)[0]
            selected_scores, rejected_scores = scores.chunk(2)
            delta = (selected_scores - rejected_scores).view(-1)
            delta = accelerator.gather_for_metrics(delta)
            all_delta_scores.extend(delta.tolist())

    if accelerator.is_main_process:
        delta_scores = np.hstack(all_delta_scores)
        accuracy = (delta_scores > 0).mean()
        #accelerator.log({"test_accuracy": accuracy})
        print(f"test accuracy: {accuracy}")
        plot_calibration("lora16_gptj_process_final", "tldr", delta_scores)


def main(
    model_id="gpt2-xl",
    batch_size=2,
    eval_batch_size=8,
    grad_accumulation=16,
    n_epochs=5,
    eval_interval=400,
    #tokenizer_id="gpt2-xl",
    tokenizer_id="EleutherAI/gpt-j-6b",
    max_seq_length=550,
    #learning_rate=3e-4,
    learning_rate=1e-5,
    weight_decay=0.0,
   # weight_decay=1e-6,
):
    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accumulation,
        log_with="wandb",
        kwargs_handlers=[kwargs]
    )
    accelerator.init_trackers(
        project_name="gptj_rm",
        init_kwargs={"wandb": {"entity": 'coactivelearning',
                               "name": 'lora16_pad'}},
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # ADD a special pad token. EOS TOKEN used for something specific
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left" # focus on the generations

    def tokenize(prompt, selected, rejected, tokenizer):
        # NOTE: for TL;DR, TL;DR: prefix is in each selected and rejected
        prompt = tokenizer.decode(
            tokenizer(
                prompt,
                truncation=True,
                max_length=500,
                add_special_tokens=False,
            )["input_ids"],
            skip_special_tokens=True,
        )
        return {
            "selected_input_ids": tokenizer(
                prompt + "\n\n" + selected + tokenizer.eos_token, truncation=True, max_length=max_seq_length
            ).input_ids,
            "rejected_input_ids": tokenizer(
                prompt + "\n\n" + rejected + tokenizer.eos_token, truncation=True, max_length=max_seq_length
            ).input_ids,
        }

    def collate_fn(batch):
        """
        batch = 16 => 32
        0 - 15 => selected
        16 - 31 => rejected
        (0, 16), (1, 17)......
        """
        #input_ids = sum([[x["selected_input_ids"], x["rejected_input_ids"]] for x in batch], [])
        input_ids = [x["selected_input_ids"] for x in batch] + [x["rejected_input_ids"] for x in batch]
        return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

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
        modules_to_save=["score"],
        task_type="SEQ_CLS", # Automatically saves scores layer
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               quantization_config=nf4_config,
                                                               num_labels=1)
    # AutoModelForCausalLM
    model.config.pad_token_id = tokenizer.pad_token_id # pad token to special token
    model.resize_token_embeddings(len(tokenizer)) # resize embeddings for pad token
    #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config=peft_config)

    # Dataset
    def reformat(sample):
        prompt = sample['prompt'].strip()
        prompt = prompt.replace("\nTITLE:", "\n\nTITLE:") \
                       .replace("\nPOST:", "\n\nPOST:")

        selected = sample["selected"].strip()
        rejected = sample["rejected"].strip()
        return {"prompt": prompt, "selected": selected, "rejected": rejected}

    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    if "chosen" in dataset["train"].column_names:
        dataset = dataset.rename_column("chosen", "selected")
    dataset = dataset.map(reformat, desc="Reformatting")
    tokenized = dataset.map(
        tokenize,
        input_columns=["prompt", "selected", "rejected"],
        fn_kwargs=dict(tokenizer=tokenizer),
        desc="Tokenizing"
    )
    train_dataloader = torch.utils.data.DataLoader(
        tokenized["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        tokenized["test"].select(range(1000)), shuffle=False, batch_size=eval_batch_size, collate_fn=collate_fn
    )

    # Optimizer
    #optimizer = AdamW(
    #    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-08, weight_decay=weight_decay
    #)
    def group_params(params, weight_decay):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [
                    p for n, p in params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return grouped_parameters
    grouped_params = group_params(model.named_parameters(), weight_decay)
    #optimizer = AdamW(
    #    grouped_params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-05, weight_decay=weight_decay
    #)
    optimizer = Adam(
        grouped_params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-08, weight_decay=weight_decay
    )
    def warmup(current_step, *, num_warmup_steps):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0
    from functools import partial
    lambda_lr = partial(warmup, num_warmup_steps=100)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    #optimizer = Adam(
    #    model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-08
    #)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3000)

    # Accelerate Prepare
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )
    #model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    #    model, optimizer, train_dataloader, val_dataloader
    #)

    step, best_accuracy = 0, 0
    tbar = tqdm(range(n_epochs * len(train_dataloader)//grad_accumulation), disable=not accelerator.is_main_process)
    for epoch in range(n_epochs):
        losses, accs, sr, rr = [], [], [], []
        for batch in train_dataloader:
            # Evaluation
            if step % eval_interval == 0:
                model.eval()
                all_delta_scores = []
                val_losses = []
                for eval_batch in tqdm(val_dataloader, desc=f"Evaluating", disable=not accelerator.is_main_process, leave=False):
                    with torch.no_grad():
                        scores = model(**eval_batch)[0]
                        selected_scores, rejected_scores = scores.chunk(2)
                        delta = (selected_scores - rejected_scores).view(-1)
                        val_loss = -F.logsigmoid(selected_scores - rejected_scores).mean()
                        delta = accelerator.gather_for_metrics(delta)
                        val_loss = accelerator.gather_for_metrics(val_loss)
                        all_delta_scores.extend(delta.tolist())
                        val_losses.extend(val_loss.tolist())
                delta_scores = np.hstack(all_delta_scores)
                accuracy = (delta_scores > 0).mean()

                accelerator.log({
                    "val_accuracy": accuracy,
                    "val_delta": delta_scores.mean().item(),
                    "val_loss": np.array(val_losses).mean().item(),
                }, step=step)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    accelerator.unwrap_model(model).save_pretrained(
                                "16_gptj_process",
                                save_function=accelerator.save,
                                is_main_process=accelerator.is_main_process,
                                state_dict=accelerator.get_state_dict(model),
                                safe_serialization=False,
                            )
                    accelerator.log({"best_accuracy": best_accuracy}, step=step)

                if accelerator.is_main_process:
                    tbar.set_postfix(accuracy=accuracy, best_accuracy=best_accuracy)

                # Set model back to train
                accelerator.wait_for_everyone()
                model.train()
            # TRIL: policies.actor.Actor <- policy with generate, get log probs, and reference (Pi_SFT)
            with accelerator.accumulate(model):
                scores = model(**batch)[0]
                selected_scores, rejected_scores = scores.chunk(2)
                #pos_labels = torch.ones_like(selected_scores)
                #neg_labels = -torch.ones_like(rejected_scores)
                #loss = F.cross_entropy(torch.cat([selected_scores, rejected_scores], dim=0), torch.cat([pos_labels, neg_labels], dim=0))
                loss = -F.logsigmoid(selected_scores - rejected_scores).mean()
                ####
                # - Log (sigmoid ( score_s - score_r)) => - (score_s - score_r) * Log (sigmoid ( (log pi(s)/ log pi_sft (s)) - (log pi(r) / log pi_sft(r))))
                ####
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Collect Training Accuracy
                train_delta = (selected_scores - rejected_scores).view(-1)
                train_delta = accelerator.gather_for_metrics(train_delta)
                batch_loss = accelerator.gather_for_metrics(loss.detach())
                selected_r = accelerator.gather_for_metrics(selected_scores.mean())
                rejected_r = accelerator.gather_for_metrics(rejected_scores.mean())

                accs.extend(train_delta.tolist())
                losses.extend(batch_loss.tolist())
                sr.extend(selected_r.tolist())
                rr.extend(rejected_r.tolist())

                if accelerator.sync_gradients:
                    # Accumulate stats
                    train_delta_scores = np.hstack(accs)
                    train_accuracy = (train_delta_scores > 0).mean()
                    train_loss = np.array(losses).mean()

                    tbar.update()
                    tbar.set_description(f"Training loss: {train_loss.item():.4f}")
                    accelerator.log({
                        "loss": train_loss.item(),
                        "train_accuracy": train_accuracy.item(),
                        "train_delta": train_delta_scores.mean().item(),
                        "selected_scores": np.array(sr).mean().item(),
                        "rejected_scores": np.array(rr).mean().item(),
                        "lr": scheduler.get_last_lr()[0]
                    }, step=step)
                    accs, losses, sr, rr = [], [], [], []

            step += 1

    # Final Eval
    for eval_batch in tqdm(val_dataloader, desc=f"Final Evaluating", disable=not accelerator.is_main_process, leave=False):
        with torch.no_grad():
            scores = model(**eval_batch)[0]
            selected_scores, rejected_scores = scores.chunk(2)
            delta = (selected_scores - rejected_scores).view(-1)
            delta = accelerator.gather_for_metrics(delta)
            all_delta_scores.extend(delta.tolist())
    delta_scores = np.hstack(all_delta_scores)
    accuracy = (delta_scores > 0).mean()

    accelerator.log({"val_accuracy": accuracy}, step=step)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        accelerator.unwrap_model(model).save_pretrained(
                    "16_gptj_process",
                    save_function=accelerator.save,
                    is_main_process=accelerator.is_main_process,
                    state_dict=accelerator.get_state_dict(model),
                )
        accelerator.log({"best_accuracy": best_accuracy}, step=step)
    accelerator.unwrap_model(model).save_pretrained(
                "16_gptj_process_final",
                save_function=accelerator.save,
                is_main_process=accelerator.is_main_process,
                state_dict=accelerator.get_state_dict(model),
            )

    accelerator.end_training()





if __name__ == '__main__':
    args = {}
    model_id = "/home/jdc396/tril/outputs/base_train_warmup_128/2023-12-06_22-20-57/model_14500"
    #main(model_id=model_id)
    #test_eval(model_id)
    get_ref(model_id)


