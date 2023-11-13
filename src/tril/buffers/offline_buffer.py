from typing import List

import jsonlines
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tril.base_task import Sample


class OfflineBuffer(Dataset):
    def __init__(self, samples: List[Sample]) -> None:
        super().__init__()
        self._samples = samples
        self._size = len(samples)

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


# TODO: make sure to allow for prefixes and suffixes for these reward model training
class PairwiseOfflineBuffer(Dataset):
    def __init__(self, samples, tokenizer, max_length=550):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(samples, desc="Processing Dataset: "):
            prompt, chosen, rejected = (
                pair.prompt_or_input_text,
                pair.chosen_text,
                pair.rejected_text,
            )
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + prompt + "\n" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + prompt + "\n" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(
                torch.eq(
                    chosen_encodings_dict["input_ids"],
                    rejected_encodings_dict["input_ids"],
                )
            ).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(
                    rejected_encodings_dict["attention_mask"]
                )

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class ExpertOfflineBuffer(Dataset):
    def __init__(self, path, tokenizer, max_length=112):
        data = list(jsonlines.open(path))

        prompts = [x["prompt_or_input_text"] for x in data]
        references = [x["reference_or_target_text"] for x in data]

        self.prompt_input_ids = []
        self.prompt_attn_masks = []
        self.reference_input_ids = []
        self.reference_attn_masks = []

        for prompt, reference in tqdm(
            zip(prompts, references), desc="Processing Dataset: "
        ):
            encodings_dict = tokenizer(
                prompt + reference,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.prompt_input_ids.append(encodings_dict["input_ids"][:, :64][0])
            self.prompt_attn_masks.append(encodings_dict["attention_mask"][:, :64][0])
            self.reference_input_ids.append(encodings_dict["input_ids"][:, -48:][0])
            self.reference_attn_masks.append(
                encodings_dict["attention_mask"][:, -48:][0]
            )

    def __len__(self):
        assert len(self.prompt_input_ids) == len(self.reference_input_ids)
        return len(self.prompt_input_ids)

    def __getitem__(self, idx):
        return (
            self.prompt_input_ids[idx],
            self.prompt_attn_masks[idx],
            self.reference_input_ids[idx],
            self.reference_attn_masks[idx],
        )


def preference_collate_fn(data):
    batch = {}
    batch["input_ids"] = torch.cat(
        [datum[0] for datum in data] + [datum[2] for datum in data]
    )
    batch["attention_mask"] = torch.cat(
        [datum[1] for datum in data] + [datum[3] for datum in data]
    )
    # batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
    return batch


def collate_fn(batch: List[Sample]):
    # dummy collate just to return the
    return batch


def create_dataloader(samples: List[Sample], batch_size: int):
    dataset = OfflineBuffer(samples)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return dataloader


def create_expert_dataloader(
    path, batch_size: int, tokenizer, shuffle=True, max_length=112
):
    dataset = ExpertOfflineBuffer(path, tokenizer, max_length)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_preference_dataloader(
    samples: List[Sample], batch_size: int, tokenizer, shuffle, max_length=550
):
    dataset = PairwiseOfflineBuffer(samples, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=preference_collate_fn,
    )
    return dataloader
