from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from peft.peft_model import set_peft_model_state_dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tril.base_reward import BaseReward, RewardType


class LearnedRewardFunction(BaseReward, nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        model_name: str,
        label_ix: int,
        include_prompt_for_eval: bool = True,
        is_trainable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(accelerator, RewardType.DIST, is_trainable)
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def get_parameters(self):
        return self.parameters()

    @torch.no_grad()
    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        prompts = []
        gens = []
        rewards = torch.zeros(len(prompt_texts))
        indices_with_done = torch.ones(len(prompt_texts)).bool()

        prompts = prompt_texts
        gens = gen_texts
        if self._include_prompt_for_eval:
            assert len(gens) == len(prompts)
            for x in range(len(gens)):
                gens[x] = prompts[x] + gens[x]

        # compute rewards at once
        encoded = self._metric_tokenizer(
            gens, return_tensors="pt", truncation=True, padding=True
        )
        outputs = self._metric_model(
            input_ids=encoded.input_ids.to(self._accelerator.device),
            attention_mask=encoded.attention_mask.to(self._accelerator.device),
        )
        scores = torch.softmax(outputs.logits, dim=1)
        scores = scores[:, self._label_ix].flatten().cpu()
        rewards[indices_with_done] = scores

        return rewards

    def forward(
        self,
        accelerator,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ):
        # NOTE: Another idea would be to just pass in tokens and policy tokenizer,
        # Decode and encode here with the appropriate tokenizers
        prompts = []
        gens = []

        prompts = prompt_texts
        gens = gen_texts
        if self._include_prompt_for_eval:
            assert len(gens) == len(prompts)
            for x in range(len(gens)):
                gens[x] = prompts[x] + gens[x]

        encoded = self._metric_tokenizer(
            gens, return_tensors="pt", truncation=True, padding=True
        )
        outputs = self._metric_model(
            input_ids=encoded.input_ids.to(accelerator.device),
            attention_mask=encoded.attention_mask.to(accelerator.device),
        )
        scores = torch.softmax(outputs.logits, dim=1)
        scores = scores[:, self._label_ix].flatten()
        return scores


class TrainableAdapterRewardFunction(BaseReward, nn.Module):
    supported_rm_modules = ("score",)

    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        adapter_id: str,
        reward_tokenizer_id: str,
        peft_config=None,
        is_trainable: bool = False,
    ):
        super().__init__(accelerator, RewardType.DIST, is_trainable)
        self.model = model

        if adapter_id:
            self.load_reward_adapter(adapter_id)
        else:
            if peft_config is None:
                raise Exception(
                    "Note you need to either pass in a pretrained adapter id or a peft config to instantiate"  # noqa
                )
            self.rm_adapter_name = "reward_adapter"
            config = LoraConfig(**peft_config)
            self.model.add_adapter(self.rm_adapter_name, config)
            hidden_dim = self.model.config.hidden_size
            self.score = nn.Linear(hidden_dim, 1).to(dtype=self.model.dtype)

        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_tokenizer_id)
        self.reward_tokenizer.pad_token = (
            self.reward_tokenizer.eos_token
        )  # TODO: properly build tokenizer

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def get_parameters(self):
        return self.parameters()

    def load_reward_adapter(
        self, adapter_model_id, adapter_name="reward_adapter", token=None
    ):
        filename = Path(adapter_model_id) / "adapter_model.bin"

        # If not local try Download
        if not filename.exists():
            try:
                local_filename = hf_hub_download(
                    adapter_model_id,
                    "adapter_model.bin",
                    token=token,
                )
            except:
                raise ValueError("Adapter not found")
        else:
            local_filename = filename

        adapter_state_dict = torch.load(local_filename, map_location="cpu")
        rm_adapter_peft_config = LoraConfig.from_pretrained(adapter_model_id)

        for score_name_candidate in self.supported_rm_modules:
            if any(
                [score_name_candidate in name for name in adapter_state_dict.keys()]
            ):
                score_name = score_name_candidate
                # we have found the correct head name and can break
                break

        score_dict = {}
        copy_adapter_state_dict = adapter_state_dict.copy()

        for name, _ in copy_adapter_state_dict.items():
            if score_name in name:
                key_name = ".".join(name.split(".")[-1:])
                score_dict[key_name] = adapter_state_dict.pop(name)

        # Add reward adapter
        self.model.add_adapter(adapter_name, rm_adapter_peft_config)
        self.rm_adapter_name = adapter_name

        num_labels, hidden_dim = score_dict["weight"].shape
        has_bias = any(["bias" in name for name in adapter_state_dict.keys()])

        # Add score layer
        self.score = nn.Linear(hidden_dim, num_labels, bias=has_bias).to(
            dtype=self.model.dtype
        )
        self.score.load_state_dict(score_dict)

        # load the adapter to the model
        set_peft_model_state_dict(
            self.model, adapter_state_dict, adapter_name=adapter_name
        )

    @torch.no_grad()
    def compute_reward(
        self,
        accelerator,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask=None,
        ref_ids=None,
        ref_mask=None,
        retokenize=True,
        scale_by_ref=False,
    ):
        self.model.set_adapter(self.rm_adapter_name)
        self.model.eval()

        # Retokenize:
        if retokenize:
            samples = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            samples = [
                "<|startoftext|>" + sample + "<|endoftext|>" for sample in samples
            ]
            encodings = self.reward_tokenizer(
                samples,
                truncation=True,
                max_length=550,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(accelerator.device)
            attention_mask = encodings["attention_mask"]

        if attention_mask is None:
            gen_attention_mask_pt = (
                input_ids.not_equal(self.reward_tokenizer.pad_token_id)
                .long()
                .to(accelerator.device)
            )
        else:
            gen_attention_mask_pt = attention_mask.to(accelerator.device)

        model_kwargs = {
            "attention_mask": gen_attention_mask_pt,
            "use_cache": False,
        }

        model_inputs = accelerator.unwrap_model(
            self.model
        ).prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_tokens_hidden = outputs.hidden_states[-1]
        rewards = self.score(last_tokens_hidden)

        # Grab rewards for last nonpad
        seq_lengths = (gen_attention_mask_pt.sum(axis=-1) - 1).long().to(rewards.device)
        rewards = rewards[
            torch.arange(input_ids.size(0), device=rewards.device), seq_lengths
        ]
        rewards = rewards.cpu()

        # Ref norm
        if ref_ids is not None and scale_by_ref:
            if retokenize:
                # Retokenize:
                samples = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                samples = [
                    "<|startoftext|>" + sample + "<|endoftext|>" for sample in samples
                ]  # TODO: make template more general
                encodings = self.reward_tokenizer(
                    samples,
                    truncation=True,
                    max_length=550,
                    padding="max_length",
                    return_tensors="pt",
                )
                ref_ids = encodings["input_ids"].to(accelerator.device)
                ref_mask = encodings["attention_mask"]
            if ref_mask is None:
                ref_attention_mask_pt = (
                    ref_ids.not_equal(self.reward_tokenizer.pad_token_id)
                    .long()
                    .to(accelerator.device)
                )
            else:
                ref_attention_mask_pt = ref_mask.to(accelerator.device)

            ref_model_kwargs = {
                "attention_mask": ref_attention_mask_pt,
                "use_cache": False,
            }
            ref_inputs = accelerator.unwrap_model(
                self.model
            ).prepare_inputs_for_generation(ref_ids, **ref_model_kwargs)
            ref_outputs = self.model(
                **ref_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_tokens_hidden = ref_outputs.hidden_states[-1]
            ref_rewards = self.score(last_tokens_hidden)

            # Grab rewards for last nonpad
            seq_lengths = (
                (ref_attention_mask_pt.sum(axis=-1) - 1).long().to(ref_rewards.device)
            )
            ref_rewards = ref_rewards[
                torch.arange(input_ids.size(0), device=ref_rewards.device), seq_lengths
            ]
            ref_rewards = ref_rewards.cpu()

            # Norm rewards with ref scores
            rewards = rewards - ref_rewards

        return rewards

    def forward(
        self,
        accelerator,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask=None,
        retokenize=True,
    ):
        self.model.set_adapter(self.rm_adapter_name)
        self.model.train()

        # Retokenize:
        if retokenize:
            samples = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            samples = [
                "<|startoftext|>" + sample + "<|endoftext|>" for sample in samples
            ]
            encodings = self.reward_tokenizer(
                samples,
                truncation=True,
                max_length=550,  # TODO: specify encoding
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(accelerator.device)
            attention_mask = encodings["attention_mask"]

        if attention_mask is None:
            gen_attention_mask_pt = (
                input_ids.not_equal(self.reward_tokenizer.pad_token_id)
                .long()
                .to(accelerator.device)
            )
        else:
            gen_attention_mask_pt = attention_mask.to(accelerator.device)

        model_kwargs = {
            "attention_mask": gen_attention_mask_pt,
            "use_cache": False,
        }

        model_inputs = accelerator.unwrap_model(
            self.model
        ).prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_tokens_hidden = outputs.hidden_states[-1]
        rewards = self.score(last_tokens_hidden)

        # Grab rewards for last nonpad
        seq_lengths = (gen_attention_mask_pt.sum(axis=-1) - 1).long().to(rewards.device)
        rewards = rewards[
            torch.arange(input_ids.size(0), device=rewards.device), seq_lengths
        ]
        return rewards
