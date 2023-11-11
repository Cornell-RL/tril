import copy
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import rouge
import torch
from accelerate import Accelerator
from datasets import load_metric
from gem_metrics_fork.msttr import MSTTR
from gem_metrics_fork.ngrams import NGramStats
from gem_metrics_fork.texts import Predictions
from transformers import PreTrainedModel

from tril.base_metric import BaseMetric
from tril.metrics.caption_metrics.cider import Cider
from tril.metrics.caption_metrics.spice.parallel_spice import ParallelSpice
from tril.metrics.caption_metrics.spice.spice import Spice

load_metric_memory = partial(load_metric, keep_in_memory=True)


class DiversityMetrics(BaseMetric):
    def __init__(self, accelerator: Accelerator, window_size: int = 100) -> None:
        super().__init__(accelerator)
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    @property
    def name(self):
        return "diversity"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        predictions = Predictions(data={"filename": "", "values": generated_texts})
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics


class MeteorMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator) -> None:
        super().__init__(accelerator)
        self._metric = load_metric_memory("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class SARIMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator) -> None:
        super().__init__(accelerator)
        self._metric = load_metric_memory("sari")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        score = self._metric.compute(
            sources=prompt_texts,
            predictions=generated_texts,
            references=reference_texts,
        )["sari"]

        metric_dict = {"lexical/sari": (None, score)}
        return metric_dict


class RougeMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator, use_single_ref: bool = True) -> None:
        super().__init__(accelerator)
        self._metric = load_metric_memory("rouge")
        self._use_single_ref = use_single_ref

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(
            predictions=generated_texts, references=ref_texts, use_stemmer=True
        )
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict


class BLEUMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator) -> None:
        super().__init__(accelerator)
        self._metric = load_metric_memory("bleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            metric_results = self._metric.compute(
                predictions=tokenized_predictions, references=tokenized_reference_texts
            )
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception:
            return {"lexical/bleu": (None, "n/a")}


class BERTScoreMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator, language: str) -> None:
        super().__init__(accelerator)
        self._metric = load_metric_memory("bertscore")
        self._language = language
        # since models are loaded heavily on cuda:0, use the last one to avoid memory
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
        predictions: Dict[str, List[str]] = None,
        references: Dict[str, List[str]] = None,
        preprocessed=False,
    ) -> Tuple[List[float], float]:
        with torch.no_grad():
            metric_results = self._metric.compute(
                predictions=generated_texts,
                references=reference_texts,
                lang=self._language,
                device=self._last_gpu,
            )
            bert_scores = metric_results["f1"]
            corpus_level_score = np.mean(bert_scores)
            metric_dict = {"semantic/bert_score": (bert_scores, corpus_level_score)}
            return metric_dict


class BLEURTMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator, config_name: str = None) -> None:
        super().__init__(accelerator)
        self._metric = load_metric_memory("bleurt", config_name=config_name)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"semantic/bleurt": (metric_results["scores"], corpus_score)}
        return metric_dict


def get_generated_and_predictions(
    prompt_texts: List[str],
    generated_texts: List[str],
    reference_texts: List[List[str]],
    split_name: str,
):
    split_name = "" if split_name is None else split_name
    preds = {}
    refs = {}
    prompt = []
    for ix, (prompt_text, gen_text, ref_text) in enumerate(
        zip(prompt_texts, generated_texts, reference_texts)
    ):
        preds[f"{ix}-" + split_name + prompt_text] = [gen_text]
        refs[f"{ix}-" + split_name + prompt_text] = ref_text
        prompt.append(f"{ix}-" + split_name + prompt_text)
    return prompt, preds, refs


def get_individual_scores(
    prompt_texts: List[str], split_name: str, scores_dict: Dict[str, float]
):
    split_name = "" if split_name is None else split_name
    scores = []
    for prompt_text in prompt_texts:
        scores.append(scores_dict.get(split_name + prompt_text, "n/a"))
    return scores


class CIDERMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator) -> None:
        super().__init__(accelerator)
        self._metric = Cider()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
        predictions: Dict[str, List[str]] = None,
        references: Dict[str, List[str]] = None,
        preprocessed=False,
        unique_prompt: List[str] = None,
    ) -> Tuple[List[float], float]:
        if preprocessed is False:
            unique_prompt, predictions, references = get_generated_and_predictions(
                prompt_texts, generated_texts, reference_texts, split_name
            )
            (
                corpus_score,
                individual_scores,
            ) = self._metric.compute_score(references, predictions)
        elif preprocessed is True:
            (
                corpus_score,
                individual_scores,
            ) = self._metric.compute_score(
                references, predictions, spacy_preprocess=True
            )

        individual_scores = get_individual_scores(
            unique_prompt, split_name, individual_scores
        )
        metric_dict = {"lexical/cider": (individual_scores, corpus_score)}
        return metric_dict


class SpiceMetric(BaseMetric):
    def __init__(
        self,
        accelerator: Accelerator,
        datapool: str = "commongen",
        role: str = "metric",
    ) -> None:
        super().__init__(accelerator)
        self._metric = Spice()
        self._parallel_metric = ParallelSpice(datapool, role)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
        predictions: Dict[str, List[str]] = None,
        references: Dict[str, List[str]] = None,
        preprocessed=False,
        unique_prompt: List[str] = None,
    ) -> Tuple[List[float], float]:
        if preprocessed is False:
            unique_prompt, predictions, references = get_generated_and_predictions(
                prompt_texts, generated_texts, reference_texts, split_name
            )

            (
                corpus_score,
                individual_scores,
            ) = self._metric.compute_score(references, predictions)
        elif preprocessed is True:
            (
                corpus_score,
                individual_scores,
            ) = self._parallel_metric.compute_score(
                references, predictions, spacy_preprocess=True
            )

        individual_scores = get_individual_scores(
            unique_prompt, split_name, individual_scores
        )

        metric_dict = {"lexical/spice": (individual_scores, corpus_score)}
        return metric_dict


class RougeLMax(BaseMetric):
    def __init__(self, accelerator: Accelerator, **args) -> None:
        super().__init__(accelerator)
        self._metric = rouge.Rouge(metrics=["rouge-l"], **args)

    def _rouge_max_over_ground_truths(self, prediction, ground_truths):
        """
        Computes max of Rouge-L (https://github.com/allenai/unifiedqa/blob/bad6ef339db6286f0d8bd0661a2daeeb0f800f59/evaluation/evaluate_narrativeqa.py#L25) # noqa
        """
        # load stemmer
        self._metric.load_stemmer(self._metric.ensure_compatibility)

        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = self._metric.get_scores(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        max_score = copy.deepcopy(score)
        max_score = max([score["rouge-l"]["f"] for score in scores_for_ground_truths])
        return max_score

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        all_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts):
            rouge_max_score = self._rouge_max_over_ground_truths(gen_text, ref_texts)
            all_scores.append(rouge_max_score)

        metric_dict = {"lexical/rouge_l_max": (all_scores, np.mean(all_scores))}
        return metric_dict


class SacreBLEUMetric(BaseMetric):
    def __init__(self, accelerator: Accelerator, **args) -> None:
        super().__init__(accelerator)
        self._args = args
        self._metric = load_metric_memory("sacrebleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["score"] / 100
        metric_dict = {"lexical/sacrebleu": (None, bleu_score)}
        return metric_dict
