from typing import Any, Dict, Optional, Union

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


class LMMultiActorCritic(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_type: str,
        model_name: str,
        guide_model_name: str,
        max_prompt_len: int,
        max_gen_len: int,
        tokenizer: PreTrainedTokenizer,
        guide_model_type: Optional[str] = None,
        beta: Union[float, Dict[str, float]] = 0.8,
        alg_type: str = "ppo_pp",
        mlp_head: bool = False,
        create_reference: bool = True,
        create_guide_critic: bool = False,
        quantize_model: bool = False,
        model: Optional[nn.Module] = None,
        peft_config: Optional[LoraConfig] = None,
        gen_kwargs: Dict[str, Any] = {},
        guide_gen_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.pad_token_id = tokenizer.pad_token_id
        self.gen_kwargs = gen_kwargs
        self.guide_gen_kwargs = guide_gen_kwargs
        self.prompt_truncation_side = prompt_truncation_side
        self.peft_config = peft_config
        self.create_reference = create_reference
        self._model_type = model_type
        self.alg_type = alg_type
        self.curr_alg_type = alg_type
        self.beta = beta

        self.create_guide_critic = create_guide_critic
        self.reference_as_guide = model_name == guide_model_name
        if self.reference_as_guide:
            assert self.create_reference

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

        if not self.reference_as_guide:
            self.guide_actor = LMActor(
                model_type=model_type if guide_model_type is None else guide_model_type,
                model_name=guide_model_name,
                max_prompt_len=max_prompt_len,
                max_gen_len=max_gen_len,
                tokenizer=tokenizer,
                create_reference=False,
                quantize_model=quantize_model,
                model=model,
                peft_config=peft_config,
                gen_kwargs=self.guide_gen_kwargs,
                prompt_truncation_side=prompt_truncation_side,
            )
            for params in self.guide_actor.parameters():
                params.requires_grad = False

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

        if self.create_guide_critic:
            self.guide_critic = LMCritic(
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
            self.guide_value_adapter_name = self.guide_critic.value_adapter_name

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

    def set_curr_alg(self, alg_type):
        self.curr_alg_type = alg_type

    def get_parameters(self):
        # Override to exclude Reference
        if self.create_guide_critic:
            return (
                list(self.actor.get_parameters())
                + list(self.critic.get_parameters())
                + list(self.guide_critic.get_parameters())
            )
        return list(self.actor.get_parameters()) + list(self.critic.get_parameters())

    def get_named_parameters(self):
        # Override to exclude Reference
        if self.create_guide_critic:
            return (
                list(self.actor.get_named_parameters())
                + list(self.critic.get_named_parameters())
                + list(self.guide_critic.get_named_parameters())
            )
        return list(self.actor.get_named_parameters()) + list(
            self.critic.get_named_parameters()
        )

    def generate(self, *args, **kwargs) -> GenerationOutput:
        # Select Beta
        if isinstance(self.beta, dict):
            beta = self.beta[self.curr_alg_type]
        else:
            beta = self.beta
        kwargs["beta"] = beta

        # Generate with proper model
        actor_fn = kwargs.pop("actor_fn")
        if actor_fn == "policy":
            return self.actor.generate(*args, **kwargs)
        elif actor_fn == "guide":
            kwargs["gen_kwargs"] = self.guide_gen_kwargs

            if self.reference_as_guide:
                kwargs["model_fn"] = "ref"
                return self.actor.generate(*args, **kwargs)
            else:
                return self.guide_actor.generate(*args, **kwargs)

    def eval_generate(self, *args, **kwargs) -> GenerationOutput:
        return self.actor.eval_generate(*args, **kwargs)

    def forward_actor(self, *args, **kwargs) -> ActorOutput:
        actor_fn = kwargs.pop("actor_fn")
        if actor_fn == "policy":
            return self.actor(*args, **kwargs)
        elif actor_fn == "ref":
            kwargs["model_fn"] = "ref"
            return self.actor(*args, **kwargs)
        elif actor_fn == "guide":
            if self.reference_as_guide:
                kwargs["model_fn"] = "ref"
                return self.actor(*args, **kwargs)
            else:
                return self.guide_actor(*args, **kwargs)

    def forward_critic(self, *args, **kwargs) -> CriticOutput:
        value_fn = kwargs.pop("value_fn")
        if value_fn == "policy":
            return self.critic(*args, **kwargs)
        elif value_fn == "guide":
            if self.create_guide_critic:
                return self.guide_critic(*args, **kwargs)
            else:
                raise NotImplementedError("Guide Critic is not created for this policy")

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
            return self.forward_actor(
                accelerator, obs=obs, actions=actions, actor_fn="policy"
            )
        if forward_critic:
            return self.forward_critic(accelerator, obs=obs, value_fn=value_fn)

        # Both
        actor_outputs = self.forward_actor(
            accelerator, obs=obs, actions=actions, actor_fn="policy"
        )
        critic_outputs = self.forward_critic(accelerator, obs=obs, value_fn=value_fn)

        # Note for FSDP, have to do a dummy pass over all the models
        if fsdp_prepare:
            ref_out = self.forward_actor(  # noqa: F841
                accelerator, obs=obs, actions=actions, actor_fn="ref"
            )
            guide_out = self.forward_actor(  # noqa: F841
                accelerator, obs=obs, actions=actions, actor_fn="guide"
            )
            if self.create_guide_critic:
                guide_critic_out = self.forward_critic(  # noqa: F841
                    accelerator, obs=obs, value_fn="guide"
                )
        return ActorCriticOutput(
            values=critic_outputs.values,
            log_prob=actor_outputs.log_probs,
            entropy=actor_outputs.entropy,
        )
