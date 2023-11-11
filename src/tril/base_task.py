from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(init=True)
class Sample:
    id: str
    prompt_or_input_text: str
    references: List[str]
    meta_data: Optional[Dict[str, Any]] = None


@dataclass(init=True)
class PreferenceSample:
    id: str
    prompt_or_input_text: str
    chosen_text: str
    rejected_text: str
    meta_data: Optional[Dict[str, Any]] = None


class BaseTask(ABC):
    """Abstract class for tasks in TRIL.

    When defining a new task in TRIL, inherit from this class and override
    the `prepare` method. There you can define how you may want to preprocess,
    format, split the dataset.
    """

    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample

    @abstractclassmethod
    def prepare(cls, **args) -> "BaseTask":
        """
        A factory method to instantiate task
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List["BaseTask"]:
        start_ix = 0
        splits = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            splits.append(type(self)(self._samples[start_ix:end_ix]))
            start_ix = end_ix
        return splits
