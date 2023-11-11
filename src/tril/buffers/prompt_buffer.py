from itertools import repeat

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PromptBuffer(Dataset):
    def __init__(
        self,
        samples,
        tokenizer,
        max_prompt_length=64,
        max_gen_length=48,
        prompt_truncation_side="left",
        context_truncation_side="right",
        prompt_padding_side="left",
        context_padding_side="right",
    ):
        self._rng = np.random.default_rng()
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_gen_length = max_gen_length
        self.prompt_truncation_side = prompt_truncation_side
        self.context_truncation_side = context_truncation_side
        self.prompt_padding_side = prompt_padding_side
        self.context_padding_side = context_padding_side

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize DB
        self.preprocess()

    def preprocess(self):
        self.db = []
        self.reference_map = {}
        for sample in tqdm(self.samples, desc="Prompt DataLoader Tokenization: "):
            # TODO: don't hardcode
            self.tokenizer.truncation_side = self.prompt_truncation_side
            self.tokenizer.padding_side = self.prompt_padding_side
            prompt_pt = self.tokenizer(
                sample.prompt_or_input_text,
                max_length=self.max_prompt_length,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )
            self.tokenizer.truncation_side = self.context_truncation_side
            self.tokenizer.padding_side = self.context_padding_side
            ref_pt = self.tokenizer(
                self._rng.choice(sample.references),
                max_length=self.max_gen_length,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )
            self.db.append(
                {
                    "prompt_or_input_encoded_pt": prompt_pt["input_ids"][0],
                    "reference_encoded_pt": ref_pt["input_ids"][0],
                }
            )

            # Create mapping for reference extraction later
            prompt = self.tokenizer.decode(
                prompt_pt["input_ids"][0], skip_special_tokens=True
            )
            self.reference_map[prompt] = sample.references

    def get_collator(self):
        def collator_fn(batch):
            self.tokenizer.truncation_side = self.prompt_truncation_side
            self.tokenizer.padding_side = self.prompt_padding_side
            prompt_out = self.tokenizer.pad(
                [{"input_ids": x["prompt_or_input_encoded_pt"]} for x in batch],
                return_tensors="pt",
                max_length=self.max_prompt_length,
                padding="max_length",
            )
            self.tokenizer.truncation_side = self.context_truncation_side
            self.tokenizer.padding_side = self.context_padding_side
            ref_out = self.tokenizer.pad(
                [{"input_ids": x["reference_encoded_pt"]} for x in batch],
                return_tensors="pt",
                max_length=self.max_gen_length,
                padding="max_length",
            )
            return {
                "prompt_or_input_encoded_pt": prompt_out["input_ids"],
                "prompt_or_input_attention_mask_pt": prompt_out["attention_mask"],
                "reference_encoded_pt": torch.cat(
                    [prompt_out["input_ids"], ref_out["input_ids"]], dim=1
                ),
                "reference_attention_mask_pt": torch.cat(
                    [prompt_out["attention_mask"], ref_out["attention_mask"]], dim=1
                ),
            }

        return collator_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.db[idx]


def create_prompt_dataloader(
    batch_size,
    samples,
    tokenizer,
    max_prompt_length,
    max_gen_length,
    prompt_truncation_side="left",
    context_truncation_side="right",
    prompt_padding_side="left",
    context_padding_side="right",
):
    dataset = PromptBuffer(
        samples,
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_gen_length=max_gen_length,
        prompt_truncation_side=prompt_truncation_side,
        context_truncation_side=context_truncation_side,
        prompt_padding_side=prompt_padding_side,
        context_padding_side=context_padding_side,
    )
    collator = dataset.get_collator()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )  # NOTE: for now
    return loader


def infinite_dataloader(loader):
    for _ in repeat(loader):
        yield from loader
