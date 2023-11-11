from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from accelerate import Accelerator
from transformers import PreTrainedModel


class MetricType(Enum):
    NON_DIST = 0  # RUNS ONLY ON MAIN PROCESS, may be sufficient for simple metrics such as Bleu, rouge etc # noqa
    DIST = 1  # RUNS ON ALL PROCESSES and results are aggregated, useful for metric which needs large models for its computation - such as Perplexity etc # noqa


class BaseMetric(ABC):
    """Abstract class for metrics used in TRIL.

    In TRIL, metrics are used for evaluation. This could be learned models
    (i.e. Pretrained reward model for RLHF or current discriminator for GAIL)
    or automated metrics METEOR, ROUGE, etc.
    """

    def __init__(
        self, accelerator: Accelerator, dist_type: MetricType = MetricType.NON_DIST
    ) -> None:
        """Init for BaseMetric.

        Args:
            accelerator: distributed computing accelerator
            dist_type: a MetricType that defines distributed behavior of the model
        """

        super().__init__()
        self._accelerator = accelerator
        self._metric_dist_type = dist_type

    @property
    def metric_type(self):
        """Attribute to get metric type."""
        return self._metric_dist_type

    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: Optional[List[Dict[str, Any]]] = None,
        model: Optional[PreTrainedModel] = None,
        split_name: Optional[str] = None,
    ):
        """Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores. # noqa

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        Args:
            prompt_texts: prompts sampled from the Task Dataset
            gen_texts: generated text to be evaluated
            ref_texts: references/labels from the Task Dataset
            meta_infos: additional components necessary for reward computation
            model: if current model is used for evaluation then pass in for inference
            split_name: split of current evaluation data
        """
        pass
