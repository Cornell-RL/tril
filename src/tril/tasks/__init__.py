from typing import Any, Dict, Type

from tril.base_task import BaseTask
from tril.tasks.tasks import IMDB, TLDR, CommonGen, IMDBForSeq2Seq, TLDRPreference


class TaskRegistry:
    _registry = {
        "imdb": IMDB,
        "commongen": CommonGen,
        "imdb_seq2seq": IMDBForSeq2Seq,
        "tldr": TLDR,
        "tldr_preference": TLDRPreference,
    }

    @classmethod
    def get(cls, task_id: str, split: str, kwargs: Dict[str, Any]) -> BaseTask:
        task_cls = cls._registry[task_id]
        task = task_cls.prepare(split=split, **kwargs)
        return task

    @classmethod
    def add(cls, id: str, task_cls: Type[BaseTask]):
        TaskRegistry._registry[id] = task_cls
