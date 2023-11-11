from typing import Any, Dict, List

from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from tril.metrics import MetricRegistry
from tril.rewards import RewardFunctionRegistry
from tril.tasks import TaskRegistry


def get_linear_fn(start: float, end: float, end_fraction: float = 1.0):
    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


def build_reward_fn(
    reward_config: Dict[str, Any], accelerator: Accelerator, model=None
):  # , is_trainable: bool=False):
    reward_config = OmegaConf.to_container(reward_config, resolve=True)
    reward_args = reward_config.get("args", {})
    reward_args["is_trainable"] = reward_args.get("is_trainable", False)
    if model is not None:
        reward_args["model"] = model
    reward_fn = RewardFunctionRegistry.get(
        reward_config["id"], accelerator, reward_args
    )
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]], accelerator: Accelerator):
    metrics = []
    for metric_config in metric_configs:
        metric_args = metric_config.get("args", {})
        metrics.append(
            MetricRegistry.get(metric_config["id"], accelerator, metric_args)
        )
    return metrics


def build_task(task_config: Dict[str, Any]):
    def _get_samples_by_split(split: str):
        kwargs = task_config.get("args", {})
        dp_split = TaskRegistry.get(task_config["id"], split, kwargs)
        return dp_split

    train_samples = _get_samples_by_split("train")
    val_samples = _get_samples_by_split("val")
    test_samples = _get_samples_by_split("test")

    samples_by_split = {
        "train": [sample for sample in train_samples],
        "val": [sample for sample in val_samples],
        "test": [sample for sample in test_samples],
    }
    return samples_by_split


def build_preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    def constant_fn(val: float):
        def func(_):
            return val

        return func

    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            _, initial_value, end_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = get_linear_fn(initial_value, end_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        elif hyperparams[key] is None:
            continue
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
    return hyperparams
