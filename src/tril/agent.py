from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    PolynomialLR,
)
from transformers import PreTrainedTokenizer

from tril.policies import PolicyRegistry
from tril.utils.builders import build_reward_fn
from tril.utils.helpers import get_optimizer_cls


class Agent(nn.Module):
    """Parent torch.nn.Module class that contains Policy(ies) and/or Reward(s)."""

    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizer,
    ):
        """Init for Agent.

        Builds all the modules defined by the cfg.
        Args:
            cfg: Full config passed in through by Hydra.
            accelerator: Accelerator for distributed computing.
            tokenizer: Task Tokenizer
        """

        super().__init__()

        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.policy_cfg = cfg.alg.policy
        self.reward_cfg = cfg.get("reward_fn", None)
        self.lora_cfg = cfg.alg.get("lora", None)

        self.max_prompt_len = cfg.sampling.max_prompt_len
        self.max_gen_len = cfg.sampling.max_gen_len

        # Instantiate Models
        self.setup_models()

        # Opimizer
        self.optimizer_cls = get_optimizer_cls(self.cfg.alg.optimizer.id)
        if self.reward_cfg is not None and self.reward_cfg.args.get(
            "is_trainable", False
        ):
            self.reward_optimizer_cls = get_optimizer_cls(self.reward_cfg.optimizer.id)

    def train(self, mode: bool) -> None:
        """Switches model between train-mode and eval-mode.

        Predominantly used to disable dropout and batchnorm.
        Args:
            mode: True for train and False for eval
        """

        self.policy.train(mode)
        if self.cfg.alg.build_reward and self.reward.is_trainable:
            self.reward.train(mode)

    # TODO unify input format. but for now
    def compute_reward(self, *args, **kwargs) -> torch.Tensor:
        """Inference pass of the reward.

        If Agent contains a reward, this method calls BaseReward compute_reward.
        For trainable rewards, this would be the inference pass.
        """

        if self.lora_cfg is not None:
            all_tokens = kwargs["all_tokens"]
            obs_tensor = kwargs["obs_tensor"]
            terminal_rewards = self.reward.compute_reward(
                self.accelerator,
                self.tokenizer,
                all_tokens,
                ref_ids=obs_tensor["reference_encoded_pt"],
                ref_mask=obs_tensor["reference_attention_mask_pt"],
            ).squeeze(-1)
        else:
            all_tokens = kwargs["all_tokens"]
            reference_map = kwargs["reference_map"]
            assert all_tokens.shape[-1] == (self.max_prompt_len + self.max_gen_len)
            prompt_tokens = all_tokens[:, : self.max_prompt_len]
            gen_tokens = all_tokens[:, -self.max_gen_len :]

            prompts = self.tokenizer.batch_decode(
                prompt_tokens, skip_special_tokens=True
            )
            gens = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            refs = [reference_map[p] for p in prompts]

            assert len(prompts) == len(gens) == len(refs)

            terminal_rewards = self.reward.compute_reward(prompts, gens, refs)
        return terminal_rewards

    @property
    def policy_params(self) -> List[torch.Tensor]:
        """Get trainable parameters from the policy."""

        return self.policy.get_parameters()

    @property
    def reward_params(self) -> List[torch.Tensor]:
        """Get trainable parameters from the reward."""

        return self.reward.get_parameters()

    @property
    def policy_named_params(self) -> List[tuple[str, torch.Tensor]]:
        """Get trainable named parameters from the policy."""

        params = list(self.policy.get_named_parameters())
        if self.lora_cfg is not None:
            adapter_params = [
                p for p in params if "adapter" in p[0] and "reward_adapter" not in p[0]
            ]
            head_params = [p for p in params if "score" in p[0]]
            params = adapter_params + head_params
        return params

    @property
    def reward_named_params(self) -> List[tuple[str, torch.Tensor]]:
        """Get trainable named parameters from the reward."""

        params = list(self.reward.named_parameters())
        if self.lora_cfg is not None:
            adapter_params = [
                p for p in params if "adapter" in p[0] and "reward_adapter" in p[0]
            ]
            score_params = [p for p in params if "score" in p[0]]
            params = adapter_params + score_params
        return params

    def setup_models(self) -> None:
        """Instantiates submodules of the agent.

        This method is called during initialization. This can setup:
        - Policy + Reward
        - Policy
        For Reward only, instantiate just a Policy == tril.policies.Critic
        """

        # Instantiate Policy
        policy_cls = PolicyRegistry.get(self.policy_cfg.id)
        policy_args = OmegaConf.to_container(self.policy_cfg.args, resolve=True)
        peft_config = None
        if self.lora_cfg is not None:
            peft_config = LoraConfig(**self.lora_cfg.peft_config)
        self.policy = policy_cls(
            tokenizer=self.tokenizer, peft_config=peft_config, **policy_args
        )
        if self.cfg.alg.build_reward:
            if self.lora_cfg is not None:
                self.reward = build_reward_fn(
                    self.reward_cfg, self.accelerator, model=self.policy.actor.model
                )
            else:
                self.reward = build_reward_fn(self.reward_cfg, self.accelerator)

    def setup_optimizer(self) -> Union[Optimizer, tuple[Optimizer, Optimizer]]:
        """Instantiates optimizers for submodules."""

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

        def create_fn(params, optim_cls, kwargs):
            grouped_params = group_params(params, kwargs.weight_decay)
            optimizer = optim_cls(grouped_params, **kwargs)
            return optimizer

        policy_optimizer = create_fn(
            self.policy_named_params, self.optimizer_cls, self.cfg.alg.optimizer.args
        )
        if not self.cfg.alg.build_reward or not self.reward.is_trainable:
            return policy_optimizer

        reward_optimizer = create_fn(
            self.reward_named_params,
            self.reward_optimizer_cls,
            self.reward_cfg.optimizer.args,
        )
        return policy_optimizer, reward_optimizer

    def create_scheduler(
        self, optimizer: Optimizer, scheduler_args: Optional[Dict[str, Any]] = None
    ) -> LRScheduler:
        """Instantiates LR Schedulers."""

        # TODO: add WARMUP stage for LR
        types = {
            "linear": PolynomialLR,
            "cosine": CosineAnnealingLR,
        }
        if scheduler_args is None or scheduler_args["id"] == "constant":
            return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)  # Constant
        decay_schedule_cls = types.get(
            scheduler_args["id"], "linear"
        )  # default to linear
        decay_scheduler = decay_schedule_cls(optimizer, **scheduler_args["args"])
        return decay_scheduler

    def forward_reward(self, *args, **kwargs):
        """Forward pass for just the reward."""

        return self.reward.forward(*args, **kwargs)

    def forward_policy(self, *args, **kwargs):
        """Forward pass for just the policy."""

        return self.policy.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Main forward method for all submodules.

        Args:
            forward_policy_only: Only do a forward pass on policy
            forward_reward_only: Only do a forward pass on reward
        """
        # Only Forward on Policy
        forward_policy_only = kwargs.pop("forward_policy_only", False)
        if forward_policy_only or not self.cfg.alg.build_reward:
            return self.forward_policy(*args, **kwargs)

        # Only Forward on Reward
        forward_reward_only = kwargs.pop("forward_reward_only", False)
        if forward_reward_only:
            assert self.reward.is_trainable
            return self.forward_reward(*args, **kwargs)

        # Forward on Both. Useful for FSDP prepare
        policy_out = self.forward_policy(*args, **kwargs)
        reward_out = None
        if self.reward.is_trainable:
            reward_out = self.forward_reward(*args, **kwargs)
        return policy_out, reward_out
