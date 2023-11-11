from functools import partial
from typing import Any, Dict, List

import numpy as np
import shortuuid
import torch
from accelerate import Accelerator
from datasets import load_metric
from nltk.stem.porter import PorterStemmer

from tril.base_reward import BaseReward
from tril.metrics.automated_metrics import (
    BERTScoreMetric,
    BLEUMetric,
    BLEURTMetric,
    CIDERMetric,
    MeteorMetric,
    RougeMetric,
    SpiceMetric,
    get_generated_and_predictions,
)
from tril.metrics.caption_metrics.spacy_preprocess import SpacyPreprocess

load_metric_memory = partial(load_metric, keep_in_memory=True)


class MeteorRewardFunction(BaseReward):
    def __init__(self, accelerator: Accelerator, **kwargs) -> None:
        super().__init__(accelerator)
        self._metric = MeteorMetric(accelerator)

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        terminal_rewards = []
        for gens, refs in zip(gen_texts, ref_texts):
            metric_dict = self._metric.compute(None, [gens], [refs])
            terminal_rewards.append(metric_dict["lexical/meteor"][1])
        return torch.tensor(terminal_rewards).float()


class RougeRewardFunction(BaseReward):
    def __init__(
        self,
        accelerator: Accelerator,
        rouge_type: str = "rouge_rouge1",
        use_single_ref: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(accelerator)
        self._metric = RougeMetric(accelerator, use_single_ref)
        self._rouge_type = rouge_type

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        terminal_rewards = []
        for gens, refs in zip(gen_texts, ref_texts):
            metric_dict = self._metric.compute(None, [gens], [refs])
            terminal_rewards.append(metric_dict[f"lexical/{self._rouge_type}"][1])
        return torch.tensor(terminal_rewards).float()


class RougeCombinedRewardFunction(BaseReward):
    def __init__(
        self, accelerator: Accelerator, use_single_ref: bool = True, **kwargs
    ) -> None:
        super().__init__(accelerator)
        self._metric = RougeMetric(accelerator, use_single_ref)

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        terminal_rewards = []
        for gens, refs in zip(gen_texts, ref_texts):
            metric_dict = self._metric.compute(prompt_texts, gen_texts, ref_texts)
            score_keys = ["rouge_rouge1", "rouge_rouge2", "rouge_rougeL"]
            scores = [metric_dict[f"lexical/{r_key}"][1] for r_key in score_keys]
            terminal_rewards.append(np.mean(scores))
        return torch.tensor(terminal_rewards).float()


class BERTScoreRewardFunction(BaseReward):
    def __init__(
        self, accelerator: Accelerator, language: str = "en", **kwargs
    ) -> None:
        super().__init__(accelerator)
        self._metric = BERTScoreMetric(accelerator, language)

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        terminal_rewards = []
        for gens, refs in zip(gen_texts, ref_texts):
            metric_dict = self._metric.compute(None, [gens], [refs])
            terminal_rewards.append(metric_dict["semantic/bert_score"][1])
        return torch.tensor(terminal_rewards).float()


class BLEURewardFunction(BaseReward):
    def __init__(self, accelerator: Accelerator, **kwargs) -> None:
        super().__init__(accelerator)
        self._metric = BLEUMetric(accelerator)

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        terminal_rewards = []
        for gens, refs in zip(gen_texts, ref_texts):
            metric_dict = self._metric.compute(None, [gens], [refs])
            terminal_rewards.append(metric_dict["lexical/bleu"][1])
        return torch.tensor(terminal_rewards).float()


class BLEURTRewardFunction(BaseReward):
    def __init__(self, accelerator: Accelerator, checkpoint: str = None, **kwargs):
        super().__init__(accelerator)
        self._metric = BLEURTMetric(accelerator, checkpoint)

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        terminal_rewards = []
        for gens, refs in zip(gen_texts, ref_texts):
            metric_dict = self._metric.compute(None, [gens], [refs])
            terminal_rewards.append(metric_dict["semantic/bleurt"][1])
        return torch.tensor(terminal_rewards).float()


class SpiderRewardFunction(BaseReward):
    def __init__(
        self,
        accelerator: Accelerator,
        spice_coeff: float,
        cider_coeff: float,
        shaping_fn: str = None,
        **kwargs,
    ) -> None:
        """
        Spice + Cider
        """
        super().__init__(accelerator)
        self._spice_metric = SpiceMetric(accelerator)
        self._cider_metric = CIDERMetric(accelerator)
        self._preprocess = SpacyPreprocess()
        self._spice_coeff = spice_coeff
        self._cider_coeff = cider_coeff

        self.su = shortuuid.ShortUUID()

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        prompts, gens, refs = [], [], []

        for ix, (prompt, gen, ref) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts)
        ):
            prompts.append("".join([prompt, f"_{self.su.random(length=8)}"]))
            gens.append(gen)
            refs.append(ref)

        if self._spice_coeff > 0 and self._cider_coeff == 0:
            spice_scores = self._spice_metric.compute(prompts, gens, refs)[
                "lexical/spice"
            ][0]
            total_scores = self._spice_coeff * torch.tensor(spice_scores)
        elif self._cider_coeff > 0 and self._spice_coeff == 0:
            cider_scores = self._cider_metric.compute(prompts, gens, refs)[
                "lexical/cider"
            ][0]
            total_scores = self._cider_coeff * torch.tensor(cider_scores)
        else:
            split_name = None
            unique_prompt, predictions, references = get_generated_and_predictions(
                prompts,
                gens,
                refs,
                split_name,
            )
            results = self._preprocess.compute_preprocess(references, predictions)

            f_spice = partial(
                self._spice_metric.compute,
                prompt_texts=prompts,
                generated_texts=gens,
                reference_texts=refs,
                preprocessed=True,
                unique_prompt=unique_prompt,
                predictions=results["res"].copy(),
                references=results["gts"].copy(),
            )

            f_cider = partial(
                self._cider_metric.compute,
                prompt_texts=prompts,
                generated_texts=gens,
                reference_texts=refs,
                preprocessed=True,
                unique_prompt=unique_prompt,
                predictions=results["res"].copy(),
                references=results["gts"].copy(),
            )

            spice_scores = f_spice()["lexical/spice"][0]
            cider_scores = f_cider()["lexical/cider"][0]

            total_scores = self._spice_coeff * torch.tensor(
                spice_scores
            ) + self._cider_coeff * torch.tensor(cider_scores)

        return total_scores


class CommonGenConceptCoverFunction(BaseReward):
    def __init__(self, accelerator: Accelerator, **kwargs) -> None:
        super().__init__(accelerator)
        self.per_step = False
        self._stemmer = PorterStemmer()

    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_info: Dict[str, Any] = None,
        per_step: bool = False,
    ) -> List[float]:
        rewards = torch.zeros(len(prompt_texts))

        for ix, (prompt, gen, ref) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts)
        ):
            prefix = "generate a sentence with: "
            concept_n_grams = prompt.split(prefix)[1].split(".")[0]
            input_concepts = concept_n_grams.lower().split(" ")
            input_concepts = set([self._stemmer.stem(x) for x in input_concepts])

            generated_concepts = gen.lower().split(" ")
            generated_concepts = set(
                [self._stemmer.stem(x) for x in generated_concepts]
            )
            concepts = input_concepts.intersection(generated_concepts)
            ratio = len(concepts) / len(input_concepts)
            # scores.append(ratio)
            rewards[ix] = ratio
        # return torch.tensor(scores)
        return rewards
