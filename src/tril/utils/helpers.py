import gc
import logging
import random
import re
from typing import List

import numpy as np
import torch
from bitsandbytes.optim import Adam8bit, AdamW8bit
from torch.optim import Adam, AdamW
from tqdm import tqdm

from tril.base_task import Sample
from tril.metrics.automated_metrics import get_generated_and_predictions
from tril.metrics.caption_metrics.spacy_preprocess import SpacyPreprocess


def print_memory(accelerator, tracemalloc, label):
    accelerator.print(f"Memory before entering {label}: {b2mb(tracemalloc.begin)}")
    accelerator.print(
        f"Memory consumed at the end of the {label} (end-begin): {tracemalloc.used}"
    )
    accelerator.print(
        f"Peak Memory consumed during the {label} (max-begin): {tracemalloc.peaked}"
    )
    accelerator.print(
        f"Total Peak Memory consumed during the {label} (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"  # noqa
    )


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    y_pred, y_true = y_pred.detach().flatten().cpu(), y_true.detach().flatten().cpu()
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)


def set_seed(seed):
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)


def get_schedule_fn(value_schedule):
    def constant_fn(val: float):
        def func(_):
            return val

        return func

    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def get_optimizer_cls(optimizer_id: str):
    try:
        optim_cls = {
            "adam": Adam,
            "adamw": AdamW,
            "adam8bit": Adam8bit,
            "adamw8bit": AdamW8bit,
        }.get(optimizer_id)
    except Exception:
        raise ValueError(
            f"{optimizer_id} is currently not supported. Please add to tril.utils.helpers."  # noqa
        )
    return optim_cls


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized. # noqa

    Args:
    - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
    - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
    Default is `[""]` to match all active loggers.
    The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *exc):
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def fsdp_prepare(agent, tokenizer, accelerator, max_length, supervised=False):
    """
    Prepare models for distributed setup
    especially for FSDP related issues
    https://github.com/huggingface/accelerate/issues/947#event-8448457764
    """
    with torch.no_grad():
        encoded = tokenizer(
            "Be happy FSDP",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        ).to(accelerator.device)

        if not supervised:
            outputs = agent.forward(
                accelerator,
                encoded["input_ids"],
                actions=None,
                fsdp_prepare=True,
                forward_policy_only=True,
            )
            if agent.reward.is_trainable:
                # TODO: use reward tokenizer, and do forward
                pass
        else:
            outputs = agent(**encoded)  # noqa: F841


def fsdp_reward_prepare(reward, accelerator):
    model = reward._metric_model
    with torch.no_grad():
        gen_text = ["FSDP prepare time"]
        encoded = reward._metric_tokenizer(
            gen_text, return_tensors="pt", truncation=True, padding=True
        )
        outputs = model(  # noqa: F841
            input_ids=encoded.input_ids.to(accelerator.device),
            attention_mask=encoded.attention_mask.to(accelerator.device),
        )


def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def preprocess_spice(spice_fn, samples_by_split, accelerator, batch_size: int = 15000):
    _preprocess_warm_start = SpacyPreprocess()
    for batch in tqdm(
        list(get_batch(samples_by_split["train"], batch_size)),
        desc="Preprocessing Spice: ",
        disable=not accelerator.is_main_process,
    ):
        refs = [sample.references for sample in batch]
        gens = [[sample.references[0]] for sample in batch]
        prompts = [sample.prompt_or_input_text for sample in batch]

        unique_prompt, predictions, references = get_generated_and_predictions(
            prompts,
            gens,
            refs,
            None,
        )

        results = _preprocess_warm_start.compute_preprocess(references, predictions)
        spice_fn.compute(
            prompt_texts=prompts,
            generated_texts=gens,
            reference_texts=refs,
            preprocessed=True,
            unique_prompt=unique_prompt,
            predictions=results["res"].copy(),
            references=results["gts"].copy(),
        )
