from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from tril.base_metric import BaseMetric, MetricType
from tril.policies.actor_critic import LMActorCritic as BasePolicy


class PreferenceRewardModelMetric(BaseMetric):
    def __init__(
        self,
        accelerator: Accelerator,
        tokenizer_id: str,
        batch_size: int,
        pad_token_as_eos_token: bool = True,
    ) -> None:
        super().__init__(accelerator, MetricType.DIST)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._batch_size = batch_size

    @property
    def name(self):
        return "preference_learned_reward"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        agent: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        model = agent
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            # NOTE: this is Summarization specific..... need to make more general
            batch_prompt_texts = [
                text.split("TL;DR:")[0] + "TL;DR: " for text in batch_prompt_texts
            ]
            input_text = [
                "<|startoftext|>" + p + "\n" + gen + "<|endoftext|>"
                for p, gen in zip(batch_prompt_texts, batch_gen_texts)
            ]

            encoded = self._tokenizer(
                input_text, return_tensors="pt", truncation=True, padding=True
            )

            obs_tensor = {
                "reference_encoded_pt": None,
                "reference_attention_mask_pt": None,
            }

            scores = self._accelerator.unwrap_model(model).compute_reward(
                all_tokens=encoded.input_ids.to(self._accelerator.device),
                obs_tensor=obs_tensor,
            )
            scores = np.array(scores.float()).tolist()
            all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/rm_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class LearnedRewardMetric(BaseMetric):
    def __init__(
        self,
        accelerator: Accelerator,
        model_name: str,
        label_ix: int,
        batch_size: int,
        include_prompt_for_eval: bool = True,
    ) -> None:
        super().__init__(accelerator, MetricType.DIST)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._accelerator.device
        )
        self._model = self._accelerator.prepare(self._model)
        self._label_ix = label_ix
        self._batch_size = batch_size
        self._include_prompt_for_eval = include_prompt_for_eval

    @property
    def name(self):
        return "learned_reward"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            if self._include_prompt_for_eval:
                batch_gen_texts = [
                    (prompt + gen)
                    for gen, prompt in zip(batch_gen_texts, batch_prompt_texts)
                ]
            encoded = self._tokenizer(
                batch_gen_texts, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoded.input_ids.to(self._accelerator.device),
                    attention_mask=encoded.attention_mask.to(self._accelerator.device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self._label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/learned_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class Perplexity(BaseMetric):
    def __init__(
        self,
        accelerator: Accelerator,
        stride: int,
        tokenizer_id: str,
        model_type: str = "causal",
        use_text_from_meta_data: bool = False,
    ) -> None:
        super().__init__(accelerator, dist_type=MetricType.DIST)
        self._tokenizer_id = tokenizer_id
        self._model_type = model_type
        self._stride = stride
        self._use_text_from_meta_data = use_text_from_meta_data
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)

    @property
    def name(self):
        return "perplexity"

    def tokenize_references(self, batch_references):
        reference_texts = [ref for refs in batch_references for ref in refs]
        self.curr_encodings = self._tokenizer(
            "\n\n".join(reference_texts), return_tensors="pt"
        )
        return self.curr_encodings.input_ids.size(-1)

    def set_batch_length(self, batch_length):
        self.batch_length = batch_length

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        agent: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        # model = agent.policy
        # TODO: clean up
        try:
            model = agent.policy.actor.model
        except:
            model = agent.model

        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        # we compute perplexity on reference texts
        if self._use_text_from_meta_data:
            reference_texts = [info["reference"] for info in meta_infos]
        else:
            reference_texts = [ref for refs in reference_texts for ref in refs]

        encodings = self._tokenizer("\n\n".join(generated_texts), return_tensors="pt")
        encodings = self.curr_encodings
        encodings.input_ids = encodings.input_ids[:, : self.batch_length]

        nlls = []

        # get model max length
        # TODO: Clean this up Kiante
        if hasattr(model, "module"):
            if isinstance(model.module, BasePolicy):
                max_length = model.get_model_max_length()
            else:
                max_length = model.module.config.n_positions
        else:
            if isinstance(model, BasePolicy):
                max_length = model.get_model_max_length()
            else:
                max_length = model.config.n_positions

        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # get inputs and target ids
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(
                self._accelerator.device
            )
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                # TODO: Clean this up Kiante
                if hasattr(model, "module"):
                    if isinstance(model.module, BasePolicy):
                        outputs = model(
                            input_ids, labels=target_ids, forward_lm_only=True
                        )
                    else:
                        outputs = model(input_ids, labels=target_ids)
                else:
                    if isinstance(model, BasePolicy):
                        outputs = model(
                            input_ids, labels=target_ids, forward_lm_only=True
                        )
                    else:
                        outputs = model(input_ids, labels=target_ids)

                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }


class OutputPerplexity(BaseMetric):
    def __init__(
        self,
        accelerator: Accelerator,
        stride: int,
        model_id: str,
        model_type: str = "causal",
    ) -> None:
        super().__init__(accelerator, dist_type=MetricType.DIST)
        self._model_id = model_id
        self._model_type = model_type
        self._stride = stride
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model = self._model.to(accelerator.device)
        self._model = accelerator.prepare(self._model)

    @property
    def name(self):
        return "output_perplexity"

    def tokenize_generations(self, generations):
        self.curr_encodings = self._tokenizer(
            "\n\n".join(generations), return_tensors="pt"
        )
        return self.curr_encodings.input_ids.size(-1)

    def set_batch_length(self, batch_length):
        self.batch_length = batch_length

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        encodings = self.curr_encodings
        encodings.input_ids = encodings.input_ids[:, : self.batch_length]

        nlls = []

        # get model max length
        max_length = self._model.config.n_positions

        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # get inputs and target ids
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(
                self._accelerator.device
            )
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self._model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/output_perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }
