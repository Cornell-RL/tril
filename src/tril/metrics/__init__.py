from typing import Any, Dict, Type

from accelerate import Accelerator

from tril.base_metric import BaseMetric
from tril.metrics.automated_metrics import (
    BERTScoreMetric,
    BLEUMetric,
    BLEURTMetric,
    CIDERMetric,
    DiversityMetrics,
    MeteorMetric,
    RougeLMax,
    RougeMetric,
    SacreBLEUMetric,
    SARIMetric,
    SpiceMetric,
)
from tril.metrics.model_metrics import (
    LearnedRewardMetric,
    OutputPerplexity,
    Perplexity,
    PreferenceRewardModelMetric,
)


class MetricRegistry:
    _registry = {
        "learned_reward": LearnedRewardMetric,
        "meteor": MeteorMetric,
        "rouge": RougeMetric,
        "bert_score": BERTScoreMetric,
        "bleu": BLEUMetric,
        "bleurt": BLEURTMetric,
        "diversity": DiversityMetrics,
        "causal_perplexity": Perplexity,
        "cider": CIDERMetric,
        "spice": SpiceMetric,
        "rouge_l_max": RougeLMax,
        "sacre_bleu": SacreBLEUMetric,
        "sari": SARIMetric,
        "causal_output_perplexity": OutputPerplexity,
        "rm_model": PreferenceRewardModelMetric,
    }

    @classmethod
    def get(
        cls, metric_id: str, accelerator: Accelerator, kwargs: Dict[str, Any]
    ) -> BaseMetric:
        metric_cls = cls._registry[metric_id]
        metric = metric_cls(accelerator, **kwargs)
        return metric

    @classmethod
    def add(cls, id: str, metric_cls: Type[BaseMetric]):
        MetricRegistry._registry[id] = metric_cls
