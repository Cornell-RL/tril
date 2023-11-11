from typing import Any, Dict, Type

from accelerate import Accelerator

from tril.base_reward import BaseReward
from tril.rewards.automated_rewards import (
    BERTScoreRewardFunction,
    BLEURewardFunction,
    BLEURTRewardFunction,
    CommonGenConceptCoverFunction,
    MeteorRewardFunction,
    RougeCombinedRewardFunction,
    RougeRewardFunction,
    SpiderRewardFunction,
)
from tril.rewards.model_rewards import (
    LearnedRewardFunction,
    TrainableAdapterRewardFunction,
)


class RewardFunctionRegistry:
    _registry = {
        "meteor": MeteorRewardFunction,
        "rouge": RougeRewardFunction,
        "bert_score": BERTScoreRewardFunction,
        "bleu": BLEURewardFunction,
        "bleurt": BLEURTRewardFunction,
        "rouge_combined": RougeCombinedRewardFunction,
        "spider": SpiderRewardFunction,
        "common_gen_concept_cover": CommonGenConceptCoverFunction,
        "learned_reward": LearnedRewardFunction,
        "adapter_reward": TrainableAdapterRewardFunction,
    }

    @classmethod
    def get(
        cls, reward_fn_id: str, accelerator: Accelerator, kwargs: Dict[str, Any]
    ) -> BaseReward:
        reward_cls = cls._registry[reward_fn_id]
        reward_fn = reward_cls(accelerator, **kwargs)
        return reward_fn

    @classmethod
    def add(cls, id: str, reward_fn_cls: Type[BaseReward]):
        RewardFunctionRegistry._registry[id] = reward_fn_cls
