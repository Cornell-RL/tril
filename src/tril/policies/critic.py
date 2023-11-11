from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import PyTorchModelHubMixin
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, PreTrainedTokenizer

from tril.utils.generation_mixin import override_generation_routines
from tril.utils.policy import AUTOMODEL_CLASS, CriticOutput, ModelType


class LMCritic(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_type: str,
        model_name: str,
        max_prompt_len: int,
        max_gen_len: int,
        tokenizer: PreTrainedTokenizer,
        mlp_head: bool = False,
        quantize_model: bool = False,
        model: Optional[nn.Module] = None,
        peft_config: Optional[LoraConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_gen_len = max_gen_len
        self.pad_token_id = tokenizer.pad_token_id
        self.peft_config = peft_config
        self._model_type = model_type

        # For Lora Models
        self.value_adapter_name = "value_adapter"

        # Init Transformer Models
        if model is None:
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

            self.model = AUTOMODEL_CLASS[model_type].from_pretrained(
                model_name,
                quantization_config=quantization_config,
            )
            self.model.__class__ = override_generation_routines(type(self.model))
            if self.peft_config is not None:
                self.model = get_peft_model(
                    self.model, self.peft_config, self.value_adapter_name
                )
        else:
            self.model = model
            if self.peft_config is not None:
                self.model.add_adapter(self.value_adapter_name, self.peft_config)

        hidden_size = self.model.config.hidden_size
        if mlp_head:
            self.score = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, 1, bias=False),
            )
        else:
            self.score = nn.Linear(hidden_size, 1, bias=False)

        if quantize_model:
            self.score = self.score.half()

    def get_parameters(self):
        if self.peft_config is not None:
            # Since we are passing by reference, self.actor.parameters gets all params
            return []
        return self.parameters()

    def get_named_parameters(self):
        if self.peft_config is not None:
            # Since we are passing by reference, self.actor.parameters gets all params
            return []
        return self.named_parameters()

    @property
    def model_type(self):
        assert self._model_type in ["causal", "seq2seq"]
        if self._model_type == "causal":
            return ModelType.CAUSAL
        else:
            return ModelType.SEQ2SEQ

    def forward(
        self,
        accelerator: Accelerator,
        obs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        return_terminal: bool = False,
    ):
        if obs is None and input_ids is None:
            raise Exception("Please define either `obs` or `input_ids`")

        self.model.eval()

        if self.peft_config is not None:
            self.model.set_adapter(self.value_adapter_name)

        input_encoded_pt = obs.int() if input_ids is None else input_ids.int()
        gen_attention_mask_pt = (
            input_encoded_pt.not_equal(self.pad_token_id).long().to(accelerator.device)
        )

        # Get Model Inputs for respective transformer type
        if self.model_type == ModelType.CAUSAL:
            model_kwargs = {
                "attention_mask": gen_attention_mask_pt,
                "use_cache": False,
            }

            model_inputs = accelerator.unwrap_model(
                self.model
            ).prepare_inputs_for_generation(input_encoded_pt, **model_kwargs)

        elif self.model_type == ModelType.SEQ2SEQ:
            # Account for Decoder Logic
            states = input_encoded_pt
            if input_encoded_pt.shape[-1] < self.max_prompt_len + self.max_gen_len:
                assert (
                    input_encoded_pt.shape[-1] + 1
                    == self.max_prompt_len + self.max_gen_len
                )
                states = torch.cat(
                    [
                        input_encoded_pt,
                        torch.zeros(
                            (input_encoded_pt.shape[0], 1),
                            dtype=input_encoded_pt.dtype,
                            device=input_encoded_pt.device,
                        ),
                    ],
                    dim=1,
                )
            encoded_pt = states[:, : self.max_prompt_len]
            decoder_input_ids = accelerator.unwrap_model(
                self.model
            ).prepare_decoder_input_ids_from_labels(states[:, -self.max_gen_len :])

            # NOTE: important for FSDP, forward with Embeddings rather than Input IDs
            embeddings = accelerator.unwrap_model(self.model).get_input_embeddings()
            input_embeds = embeddings(encoded_pt.int())
            decode_embeds = embeddings(decoder_input_ids.int().to(accelerator.device))

            model_inputs = {
                "inputs_embeds": input_embeds,
                "decoder_inputs_embeds": decode_embeds,
                "use_cache": True,
            }

        outputs = accelerator.unwrap_model(self.model).forward(
            **model_inputs,
            output_hidden_states=True,
        )

        if self.model_type == ModelType.CAUSAL:
            last_tokens_hidden = outputs.hidden_states[-1][:, -self.max_gen_len :, :]
        else:
            last_tokens_hidden = outputs.decoder_hidden_states[-1]
        traj_shape = (input_encoded_pt.shape[0], self.max_gen_len)  # batch, seq
        values = self.score.forward(
            last_tokens_hidden.reshape(-1, self.model.config.hidden_size)
        )

        # This is useful for Reward Training where we want R(S_H)
        if return_terminal:
            seq_lengths = (
                (gen_attention_mask_pt[:, -self.max_gen_len].sum(axis=-1) - 1)
                .long()
                .to(values.device)
            )
            values = values[
                torch.arange(input_encoded_pt.size(0), device=values.device),
                seq_lengths,
            ]
            return values

        # This is useful for Value/Critic Training where we want V(S_t) for t in [0, H]
        values = values.reshape(*traj_shape)
        return CriticOutput(values.float())
