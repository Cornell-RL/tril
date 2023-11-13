from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import PyTorchModelHubMixin
from numpy.random import Generator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributions import Categorical
from transformers import BitsAndBytesConfig, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from tril.utils.generation_mixin import override_generation_routines
from tril.utils.logit_processors import RollinProcessor
from tril.utils.policy import AUTOMODEL_CLASS, ActorOutput, GenerationOutput, ModelType


class LMActor(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_type: str,
        model_name: str,
        max_prompt_len: int,
        max_gen_len: int,
        tokenizer: PreTrainedTokenizer,
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

        # For Lora Models
        self.policy_adapter_name = "policy_adapter"

        # Init Transformer Models
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if quantize_model
            else None
        )

        if model is None:
            self.model = AUTOMODEL_CLASS[model_type].from_pretrained(
                model_name,
                quantization_config=quantization_config,
            )
            self.model.__class__ = override_generation_routines(type(self.model))
            if self.peft_config is not None:
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=False
                )  # TODO: flag for gradient checkpointing
                self.model = get_peft_model(
                    self.model, self.peft_config, self.policy_adapter_name
                )
        else:
            self.model = model
            if self.peft_config is not None:
                self.model.add_adapter(self.policy_adapter_name, self.peft_config)

        # Don't create Separate Model for Lora Based Models
        if self.create_reference and self.peft_config is None:
            self.ref_model = AUTOMODEL_CLASS[model_type].from_pretrained(
                model_name,
                quantization_config=quantization_config,
            )
            self.ref_model.__class__ = override_generation_routines(
                type(self.ref_model)
            )
            if model_type == "causal":
                for param in self.ref_model.parameters():
                    param.requires_grad = False
            self.ref_model.eval()

    @property
    def model_type(self):
        assert self._model_type in ["causal", "seq2seq"]
        if self._model_type == "causal":
            return ModelType.CAUSAL
        else:
            return ModelType.SEQ2SEQ

    def get_parameters(self):
        # Override to exclude Reference
        return self.model.parameters()

    def get_named_parameters(self):
        # Override to exclude Reference
        return self.model.named_parameters()

    def get_model(self, model_fn):
        if not self.create_reference or self.peft_config is not None:
            return self.model
        if model_fn == "policy":
            return self.model
        elif model_fn == "ref":
            return self.ref_model
        else:
            raise NotImplementedError(
                f"{model_fn} is not an existing model in this actor"
            )

    def get_context_manager(self, model_fn):
        if not self.create_reference or self.peft_config is None:
            return nullcontext()

        if model_fn == "policy":
            return nullcontext()
        elif model_fn == "ref":
            return self.model.disable_adapter()
        else:
            raise NotImplementedError(
                f"{model_fn} is not an existing model in this actor"
            )

    def forward(
        self,
        accelerator: Optional[Accelerator] = None,
        obs: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        model_fn: str = "policy",
    ):
        if obs is None and input_ids is None:
            raise Exception("Please define either `obs` or `input_ids`")

        # Grabs either Actor Model or Reference Model
        model = self.get_model(model_fn)
        model.eval()

        # If lora based, set active
        if self.peft_config is not None:
            model.set_adapter(self.policy_adapter_name)

        # Disables Adapters if Lora Policy and Reference Model
        cm = self.get_context_manager(model_fn)
        with cm:
            # When defining with `input_ids` we are going through HF transformers forward. # noqa
            # Useful when computing loss through HF for supervised pipelines
            if input_ids is not None and attention_mask is not None:
                # For FSDP prepare purposes
                if labels is None:
                    labels = input_ids.clone()

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if self.model_type == ModelType.SEQ2SEQ:
                    model_inputs["decoder_input_ids"] = decoder_input_ids
                outputs = model(**model_inputs)
                return outputs

            # When doing forward with Obs, we are in Online/RL Pipelines
            input_encoded_pt = obs.int()
            # Get Model Inputs for respective transformer type
            if self.model_type == ModelType.CAUSAL:
                gen_attention_mask_pt = (
                    input_encoded_pt.not_equal(self.pad_token_id)
                    .long()
                    .to(accelerator.device)
                )
                model_kwargs = {
                    "attention_mask": gen_attention_mask_pt,
                    "use_cache": False,
                }

                model_inputs = accelerator.unwrap_model(
                    model
                ).prepare_inputs_for_generation(input_encoded_pt, **model_kwargs)

            elif self.model_type == ModelType.SEQ2SEQ:
                input_encoded_pt = input_encoded_pt[:, : self.max_prompt_len]
                if actions is not None:
                    decoder_input_ids = accelerator.unwrap_model(
                        model
                    ).prepare_decoder_input_ids_from_labels(actions)
                else:
                    decoder_encoded_pt = self.tokenizer(
                        ["" for _ in range(obs.shape[0])],
                        padding="max_length",
                        max_length=self.max_gen_len,
                        truncation=True,
                        return_tensors="pt",
                    )
                    decoder_input_ids = accelerator.unwrap_model(
                        model
                    ).prepare_decoder_input_ids_from_labels(
                        decoder_encoded_pt["input_ids"]
                    )
                # NOTE: important for FSDP, foward with Embeddings rather than Input IDs
                embeddings = accelerator.unwrap_model(model).get_input_embeddings()
                input_embeds = embeddings(input_encoded_pt.int())
                decode_embeds = embeddings(
                    decoder_input_ids.int().to(accelerator.device)
                )

                model_inputs = {
                    "inputs_embeds": input_embeds,
                    "decoder_inputs_embeds": decode_embeds,
                    "use_cache": True,
                }

            outputs = accelerator.unwrap_model(model).forward(
                **model_inputs,
                output_hidden_states=False,
            )

            # Returns Dummy Output. Useful for FSDP preparation
            if actions is None:
                return ActorOutput(None, None)

            next_token_logits = outputs.logits[:, -self.max_gen_len :, :]
            dist = Categorical(logits=next_token_logits)
            entropy = dist.entropy()
            log_prob = dist.log_prob(actions)

            # Returns Policy Distribution outputs
            return ActorOutput(log_prob, entropy)

    def generate(
        self,
        accelerator: Accelerator,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        model_fn: str = "policy",
        beta: Optional[float] = None,
        rng: Optional[Generator] = None,
        rollin_actions: Optional[torch.Tensor] = None,
        rollin_seq_lens: Optional[List[int]] = None,
        return_mask: bool = False,
    ):
        # Grabs either Actor Model or Reference Model
        model = self.get_model(model_fn)

        if self.peft_config is not None:
            model.set_adapter(self.policy_adapter_name)

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            if isinstance(self.gen_kwargs, DictConfig):
                gen_kwargs = OmegaConf.to_container(self.gen_kwargs, resolve=True)
            else:
                gen_kwargs = self.gen_kwargs

        # switch to eval
        model.eval()

        # instantiate logit processors
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )

        # Logic: (1) no rollin actions => rollins with no teacher forcing
        #        (2) rollin actions => rollouts with teacher forcing

        if rollin_actions is not None:
            logits_processor.append(
                RollinProcessor(rollin_actions, beta, rng, rollin_seq_lens)
            )

        # if min_length argument is set and if policy is not seq2seq (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if "min_length" in gen_kwargs.keys() and self.model_type != ModelType.SEQ2SEQ:
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_["min_length"] = (
                input_ids.shape[1] + gen_kwargs["min_length"]
            )
        else:
            generation_kwargs_ = deepcopy(gen_kwargs)

        # Disables Adapters if Lora Policy and Reference Model
        cm = self.get_context_manager(model_fn)
        with cm:
            # generate
            gen_output = accelerator.unwrap_model(self.model).generate(
                inputs=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                return_dict_in_generate=True,
                output_scores=True,
                synced_gpus=False,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id
                if self.model_type == ModelType.CAUSAL
                else self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs_,
            )
            mask = (
                None
                if rollin_actions is None
                else logits_processor[0].get_rollin_mask()
            )
            if return_mask:
                return gen_output, mask
            return gen_output

    def eval_generate(
        self,
        tokenizer: PreTrainedTokenizer,
        accelerator: Accelerator,
        texts: Optional[List[str]] = None,
        sample_ids: Optional[torch.Tensor] = None,
        max_prompt_length: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        gather_from_devices: bool = True,
    ) -> GenerationOutput:
        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self.gen_kwargs

        # switch to eval
        self.model.eval()
        if self.peft_config is not None:
            self.model.set_adapter(self.policy_adapter_name)

        if (
            input_ids is None
            and attention_mask is None
            and texts is not None
            and max_prompt_length is not None
        ):
            # override truncation side for prompt
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self.prompt_truncation_side
            encodings = tokenizer(
                texts,
                padding="max_length",
                max_length=max_prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            tokenizer.truncation_side = prev_truncation_side

        # if min_length argument is set and if policy is not seq2seq (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if "min_length" in gen_kwargs.keys() and self.model_type != ModelType.SEQ2SEQ:
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_["min_length"] = (
                input_ids.shape[1] + gen_kwargs["min_length"]
            )
        else:
            generation_kwargs_ = gen_kwargs
        # generate
        gen_output = accelerator.unwrap_model(self.model).generate(
            inputs=input_ids.to(accelerator.device),
            attention_mask=attention_mask.to(accelerator.device),
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            synced_gpus=False,
            pad_token_id=tokenizer.eos_token_id
            if self.model_type == ModelType.CAUSAL
            else tokenizer.pad_token_id,
            **generation_kwargs_,
        )

        # number of tokens generated
        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]

        if gather_from_devices:
            # now we got to pad the gen_tokens to maximum sequence length
            max_length = generation_kwargs_["max_new_tokens"]  # TBD: fix this
            if seq_length < max_length:
                prev_padding_side = tokenizer.padding_side
                tokenizer.padding_side = "right"
                padded_gen_tokens = tokenizer.pad(
                    {"input_ids": gen_tokens},
                    padding="max_length",
                    max_length=max_length,
                )["input_ids"].to(accelerator.device)
                tokenizer.padding_size = prev_padding_side
            else:
                padded_gen_tokens = gen_tokens.to(accelerator.device)
            gathered_gen_tokens = accelerator.gather_for_metrics(padded_gen_tokens)

            gathered_gen_texts = []
            for output in gathered_gen_tokens.tolist():
                text = tokenizer.decode(output, skip_special_tokens=True)
                gathered_gen_texts.append(text)

            gathered_sample_ids = accelerator.gather_for_metrics(sample_ids).tolist()
            assert len(gathered_gen_texts) == len(gathered_sample_ids)

            generation_outputs = GenerationOutput(
                None, None, None, gathered_gen_texts, gathered_sample_ids
            )
            return generation_outputs
        else:
            gathered_sample_ids = None

        gen_texts = []
        for output in gen_tokens.tolist():
            text = tokenizer.decode(output, skip_special_tokens=True)
            gen_texts.append(text)

        # extract scores (logits)
        step_wise_logprobs = []
        step_wise_actions = []
        for step, logits in enumerate(gen_output["scores"]):
            actions_at_step = gen_tokens[:, step]
            step_wise_logprobs.append(None)
            step_wise_actions.append(actions_at_step)

        gen_output = GenerationOutput(
            step_wise_logprobs,
            step_wise_actions,
            gen_tokens,
            gen_texts,
            gathered_sample_ids,
        )
        return gen_output
