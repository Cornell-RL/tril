from abc import ABC, abstractclassmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from accelerate import Accelerator


class RewardType(Enum):
    """Enum for distributed or non-distributed reward typing.

    NON_DIST Runs only on main process
    DIST Runs on all processes
    """

    NON_DIST = 0
    DIST = 1


class BaseReward(ABC):
    """Abstract class for all reward functions used in TRIL.

    In TRIL, a reward could be either trainble and/or distributed.
    Examples:
        1) RL pipelines that use an automated metric such as Rouge for a reward, the reward is not # noqa
        trainable and is not distributed.
        2) RL pipelines that use learned reward (i.e. RLHF), the reward is not trainable but we may want # noqa
        to make use of distributed inference for faster inference over large model.
        3) IL pipelines with learned reward (i.e. GAIL), the reward is trainable and depending on model size # noqa
        we may or may not want to do distributed inference/training.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        dist_type: RewardType = RewardType.NON_DIST,
        is_trainable: bool = False,
    ) -> None:
        """Init for BaseReward.

        Args:
            accelerator: distributed computing accelerator
            dist_type: a RewardType that defines distributed behavior of the model
            is_trainable: Whether or not reward is trainable (i.e. Imitation Learning) or not (i.e. Reinforcement Learning) # noqa
        """

        super().__init__()
        self._accelerator = accelerator
        self._dist_type = dist_type
        self._is_trainable = is_trainable

    @property
    def is_trainable(self) -> bool:
        """Flag to define whether reward is trainable or not."""
        return self._is_trainable

    @abstractclassmethod
    def compute_reward(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        meta_infos: Optional[List[Dict[str, Any]]] = None,
    ):
        """Inference pass to do batch computation of the reward.

        Args:
            prompt_texts: prompts sampled from the Task Dataset
            gen_texts: generated text to be evaluated
            ref_texts: references/labels from the Task Dataset
            meta_infos: additional components necessary for reward computation
        """
        pass
