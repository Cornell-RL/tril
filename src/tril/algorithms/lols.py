from typing import Optional

import numpy as np
from accelerate import Accelerator

from tril.algorithms.aggrevated import AGGREVATED
from tril.algorithms.ppo_pp import PPO_PP
from tril.logging import Tracker


class LOLS(PPO_PP, AGGREVATED):
    def __init__(
        self,
        cfg,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        super().__init__(cfg, accelerator, tracker)

        self.rng = np.random.default_rng(seed=self.seed + 123)
        self.rng2 = np.random.default_rng(seed=self.seed + 12345)

        self.accelerator.wait_for_everyone()

    def choose_alg(self):
        self.value_fn = "policy" if self.rng2.random() < self.tau else "guide"
        alg_type = "ppo_pp" if self.value_fn == "policy" else "aggrevate"
        return alg_type

    def collect_rollouts(self):
        # Sample which rollin/rollout scheme to play
        alg_type = self.choose_alg()
        self.agent.policy.set_curr_alg(alg_type)
        self.rollin_policy = (
            "policy" if self.agent.policy.curr_alg_type == "aggrevate" else "guide"
        )
        self.rollout_policy = (
            "policy" if self.agent.policy.curr_alg_type == "ppo_pp" else "guide"
        )

        if alg_type == "ppo_pp":
            PPO_PP.collect_rollouts(self)
        else:
            AGGREVATED.collect_rollouts(self)
