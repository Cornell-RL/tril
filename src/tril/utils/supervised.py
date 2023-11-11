from datasets.arrow_dataset import Dataset

from tril.base_task import BaseTask


def get_datasets_for_causal(train_dataset: BaseTask):
    texts = []
    for sample in train_dataset:
        for ref in sample.references:
            text = sample.prompt_or_input_text + ref
            texts.append(text)

    train_dataset = Dataset.from_dict({"content": texts}, split="train")
    return train_dataset


def get_datasets_for_seq2seq(train_dataset: BaseTask):
    articles = []
    summaries = []
    for sample in train_dataset:
        for ref in sample.references:
            articles.append(sample.prompt_or_input_text)
            summaries.append(ref)

    train_dataset = Dataset.from_dict(
        {"input_text": articles, "output_text": summaries}, split="train"
    )
    return train_dataset


def tokenize_causal(item, tokenizer):
    outputs = tokenizer(
        item["content"],
        truncation=True,
    )
    return {"input_ids": outputs["input_ids"]}


def tokenize_seq2seq(item, tokenizer):
    model_inputs = tokenizer(
        item["input_text"],
        truncation=True,
    )
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(item["output_text"], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
