from functools import wraps
from typing import Callable

from omegaconf import DictConfig


def tril_run(func: Callable):
    @wraps(func)
    def decorator(cfg: DictConfig):
        task = cfg.task.id

        # Check for task/algorithm configuration
        alg_id = cfg.alg.alg_id
        try:
            alg_config = cfg.alg[task]
        except Exception:
            raise NotImplementedError(
                f"There is no respective config for {task} in {alg_id}. Please add to cfgs/alg/{alg_id}.yaml"  # noqa
            )
        if alg_id != alg_config.id:
            raise NotImplementedError(
                f"There is no respective config for {task} in {alg_id}. Please add to cfgs/alg/{alg_id}.yaml"  # noqa
            )

        # Allow for Algorithms to override Task Reward configs.
        # Useful for Imitation Learning setups
        if hasattr(alg_config, "reward_fn"):
            cfg.reward_fn = alg_config.reward_fn
            delattr(alg_config, "reward_fn")
        cfg.alg = alg_config
        return func(cfg)

    return decorator
