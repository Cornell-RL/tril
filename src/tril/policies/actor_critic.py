from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import PyTorchModelHubMixin
from peft import LoraConfig
from transformers import PreTrainedTokenizer

from tril.policies.actor import LMActor
from tril.policies.critic import LMCritic
from tril.utils.policy import (
    ActorCriticOutput,
    ActorOutput,
    CriticOutput,
    GenerationOutput,
    ModelType,
)


class LMActorCritic(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_type: str,
        model_name: str,
        max_prompt_len: int,
        max_gen_len: int,
        tokenizer: PreTrainedTokenizer,
        mlp_head: bool = False,
        create_reference: bool = True,
        quantize_model: bool = False,
        model: Optional[nn.Module] = None,
        peft_config: Optional[LoraConfig] = None,
        gen_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.pad_token_id = tokenizer.pad_token_id
        self.gen_kwargs = gen_kwargs
        self.prompt_truncation_side = prompt_truncation_side
        self.peft_config = peft_config
        self.create_reference = create_reference
        self._model_type = model_type
        self.curr_alg_type = "ppo"

        # Create Models
        self.actor = LMActor(
            model_type=model_type,
            model_name=model_name,
            max_prompt_len=max_prompt_len,
            max_gen_len=max_gen_len,
            tokenizer=tokenizer,
            create_reference=create_reference,
            quantize_model=quantize_model,
            model=model,
            peft_config=peft_config,
            gen_kwargs=gen_kwargs,
            prompt_truncation_side=prompt_truncation_side,
        )
        if self.peft_config is not None:
            # For LORA, we want Critic to add an adapter to base model
            model = self.actor.model

        self.critic = LMCritic(
            model_type=model_type,
            model_name=model_name,
            max_prompt_len=max_prompt_len,
            max_gen_len=max_gen_len,
            tokenizer=tokenizer,
            mlp_head=mlp_head,
            quantize_model=quantize_model,
            model=model,
            peft_config=peft_config,
        )

        # For Lora Models
        self.policy_adapter_name = self.actor.policy_adapter_name
        self.value_adapter_name = self.critic.value_adapter_name

    @property
    def model_type(self) -> ModelType:
        assert self._model_type in ["causal", "seq2seq"]
        if self._model_type == "causal":
            return ModelType.CAUSAL
        else:
            return ModelType.SEQ2SEQ

    def get_model_max_length(self):
        return self.actor.model.config.n_positions

    def get_parameters(self) -> List[torch.Tensor]:
        # Override to exclude Reference
        return list(self.actor.get_parameters()) + list(self.critic.get_parameters())

    def get_named_parameters(self) -> List[tuple[str, torch.Tensor]]:
        # Override to exclude Reference
        return list(self.actor.get_named_parameters()) + list(
            self.critic.get_named_parameters()
        )

    def generate(self, *args, **kwargs) -> GenerationOutput:
        return self.actor.generate(*args, **kwargs)

    def eval_generate(self, *args, **kwargs) -> GenerationOutput:
        return self.actor.eval_generate(*args, **kwargs)

    def forward_actor(self, *args, **kwargs) -> ActorOutput:
        return self.actor(*args, **kwargs)

    def forward_critic(self, *args, **kwargs) -> CriticOutput:
        return self.critic(*args, **kwargs)

    def forward(
        self,
        accelerator: Accelerator,
        obs: torch.Tensor,
        actions: torch.Tensor,
        forward_actor: bool = False,
        forward_critic: bool = False,
        fsdp_prepare: bool = False,
        value_fn: str = "policy",
    ) -> ActorCriticOutput:
        # Individual forwards for training
        if forward_actor:
            return self.forward_actor(accelerator, obs=obs, actions=actions)
        if forward_critic:
            return self.forward_critic(accelerator, obs=obs)

        # Both
        actor_outputs = self.forward_actor(accelerator, obs=obs, actions=actions)
        critic_outputs = self.forward_critic(accelerator, obs=obs)

        # Note for FSDP, have to do a dummy pass over all the models
        if fsdp_prepare:
            ref_out = self.forward_actor(  # noqa: F841
                accelerator, obs=obs, actions=actions, model_fn="ref"
            )
        return ActorCriticOutput(
            values=critic_outputs.values,
            log_prob=actor_outputs.log_probs,
            entropy=actor_outputs.entropy,
        )
