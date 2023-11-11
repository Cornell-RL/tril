from typing import Type

import torch

from tril.policies.actor import LMActor
from tril.policies.actor_critic import LMActorCritic
from tril.policies.critic import LMCritic
from tril.policies.multi_actor_critic import LMMultiActorCritic


class PolicyRegistry:
    _registry = {
        "actor": LMActor,
        "critic": LMCritic,
        "reward": LMCritic,
        "actor_critic": LMActorCritic,
        "multi_actor_critic": LMMultiActorCritic,
    }

    @classmethod
    def get(cls, policy_id: str) -> Type[torch.nn.Module]:
        policy_cls = cls._registry[policy_id]
        return policy_cls

    @classmethod
    def add(cls, id: str, policy_cls: Type[torch.nn.Module]):
        PolicyRegistry._registry[id] = policy_cls
