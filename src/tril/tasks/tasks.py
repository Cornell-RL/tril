from typing import Dict

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tril.base_task import BaseTask, PreferenceSample, Sample


class CommonGen(BaseTask):
    @classmethod
    def prepare(
        cls,
        split: str,
        concept_separator_token: str = " ",
        concept_end_token=" ",
        prefix: str = "summarize: ",
    ) -> "BaseTask":
        ds = load_dataset("gem", "common_gen")
        samples = []
        split_id = CommonGen.gen_split_name(split)
        samples_grouped_by_concepts = {}
        for ix, item in enumerate(ds[split_id]):
            concepts = concept_separator_token.join(item["concepts"])
            concepts = prefix + concepts
            concepts += concept_end_token

            # get the references
            if len(item["references"]) > 0:
                # use the reference
                targets = item["references"]
            else:
                # otherwise use the target
                if len(item["target"]) == 0:
                    # just to avoid breaking of metric computation
                    targets = ["empty reference"]
                else:
                    targets = [item["target"]]

            sample = Sample(
                id=ix,
                prompt_or_input_text=concepts,
                references=targets,
                meta_data={"concepts": item["concepts"]},
            )
            if concepts not in samples_grouped_by_concepts.keys():
                samples_grouped_by_concepts[concepts] = sample
            else:
                samples_grouped_by_concepts[concepts].references.extend(
                    sample.references
                )

        # create samples
        samples = list(samples_grouped_by_concepts.values())

        task_instance = cls(samples)
        return task_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class TLDR(BaseTask):
    @classmethod
    def prepare(
        cls,
        split: str,
        tokenizer_id: str,
        max_prompt_length: int,
        n_samples: Dict[str, int] = {"valid": 100, "test": 500},
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id
        )  # NOTE: truncation side | right, padding side | left
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "right"

        def process_prompts(example, idxs):
            prompt = example["prompt"]
            processed_prompt = [p.split("TL;DR:")[0] for p in prompt]
            tmp = tokenizer.batch_decode(
                tokenizer(
                    processed_prompt,
                    truncation=True,
                    max_length=max_prompt_length
                    - 5,  # to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            )
            tmp = [t.strip() + "\nTL;DR:" for t in tmp]
            tmp = tokenizer.batch_decode(
                tokenizer(
                    tmp,
                    truncation=True,
                    max_length=max_prompt_length,
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            )
            tmp = [t.strip() for t in tmp]
            return {"id": idxs, "prompt": tmp, "label": example["label"]}

        ds = load_dataset("CarperAI/openai_summarize_tldr")
        split_name = TLDR.gen_split_name(split)
        samples = []

        # Map does caching
        split_ds = ds[split_name].map(
            process_prompts, with_indices=True, batched=True, batch_size=1000
        )
        n_split = n_samples.get(split_name, len(split_ds))
        for prompt, label, ids in tqdm(
            zip(
                split_ds[:n_split]["prompt"],
                split_ds[:n_split]["label"],
                split_ds[:n_split]["id"],
            ),
            desc=f"Preprocessing {split} Prompts",
            total=n_split,
        ):
            # Create Sample
            sample = Sample(
                id=ids,
                prompt_or_input_text=prompt,
                references=[label],
                meta_data={"reference": label},
            )
            samples.append(sample)
        task_instance = cls(samples)
        return task_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "valid"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class TLDRPreference(BaseTask):
    @classmethod
    def prepare(cls, split: str):
        ds = load_dataset("CarperAI/openai_summarize_comparisons")
        split_name = TLDRPreference.gen_split_name(split)
        samples = []
        for ix, item in enumerate(ds[split_name]):
            if item["chosen"] == item["rejected"]:
                continue
            if len(item["chosen"].split()) < 5 or len(item["rejected"].split()) < 5:
                continue

            sample = PreferenceSample(
                id=ix,
                prompt_or_input_text=item["prompt"],
                chosen_text=item["chosen"],
                rejected_text=item["rejected"],
                meta_data={},
            )
            samples.append(sample)
        task_instance = cls(samples)
        return task_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "valid1"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class IMDB(BaseTask):
    """
    IMDB Dataset for sentiment continuation task
    """

    @classmethod
    def prepare(cls, split: str, seed: int):
        dataset = load_dataset("imdb", ignore_verifications=True)
        if split in ["train", "val"]:
            dataset_split = dataset["train"].shuffle(
                seed=seed
            )  # Set to match with expert for every run
            train_ratio = 0.8
            train_index = int(len(dataset_split) * train_ratio)
            dataset_split = (
                dataset_split[:train_index]
                if split == "train"
                else dataset_split[train_index:]
            )
        else:
            dataset_split = dataset[split].shuffle(seed=seed)
            dataset_split = dataset_split[:5000]

        samples = []
        for ix, text in enumerate(dataset_split["text"]):
            # here we consider 50% of tokens as prompt
            prompt_text = text.split(" ")
            prompt_text = " ".join(prompt_text[: int(len(prompt_text) * 0.5)])

            sample = Sample(
                id=ix,
                prompt_or_input_text=prompt_text,
                references=[text],
                meta_data={"reference": text},
            )
            samples.append(sample)
        task_instance = cls(samples)
        return task_instance


class IMDBForSeq2Seq(BaseTask):
    """
    IMDB Dataset in seq2seq format to train supervised generator
    """

    @classmethod
    def prepare(cls, split: str, seed: int, positive_ratio: int = 1.0):
        dataset = load_dataset("imdb")
        if split in ["train", "val"]:
            dataset_split = dataset["train"].shuffle(seed=seed)
            train_ratio = 0.8
            train_index = int(len(dataset_split) * train_ratio)
            dataset_split = (
                dataset_split[:train_index]
                if split == "train"
                else dataset_split[train_index:]
            )
        else:
            # limit test to 5000
            dataset_split = dataset[split].shuffle(seed=seed)
            dataset_split = dataset_split[:5000]

        samples = []
        for ix, (text, label) in enumerate(
            zip(dataset_split["text"], dataset_split["label"])
        ):
            # here we consider 50% of tokens as prompt and rest as references
            tokenized_text = text.split(" ")
            text_split_index = int(len(tokenized_text) * 0.5)
            prompt_text = " ".join(tokenized_text[:text_split_index])
            ref_text = " ".join(tokenized_text[text_split_index:])

            # add only positive examples for train set
            # if split == "train" and label == 1 or split != "train":
            if label == 1:
                # import pdb; pdb.set_trace()
                sample = Sample(
                    id=f"{split}_{ix}",
                    prompt_or_input_text=prompt_text,
                    references=[ref_text],
                    meta_data={"reference": text},
                )
                samples.append(sample)

        # truncate train split
        if split == "train":
            samples = samples[: int(len(samples) * positive_ratio)]

        task_instance = cls(samples)
        return task_instance
