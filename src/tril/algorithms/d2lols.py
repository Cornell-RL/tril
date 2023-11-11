from typing import Optional

from accelerate import Accelerator

from tril.algorithms.lols import LOLS
from tril.logging import Tracker


class D2LOLS(LOLS):
    def __init__(
        self,
        cfg,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        super().__init__(cfg, accelerator, tracker)

    def choose_alg(self):
        self.value_fn = "policy" if self.iteration >= self.tau else "guide"
        alg_type = "ppo_pp" if self.value_fn == "policy" else "aggrevate"
        return alg_type
