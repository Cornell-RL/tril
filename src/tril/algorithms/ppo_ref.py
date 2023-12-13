from typing import Any, Dict, List, Tuple, Type, Optional, Union

import gc
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from tril.logging import Tracker
from tril.algorithms.ppo_pp import PPO_PP

class PPOReference(PPO_PP):
    def __init__(self, cfg, accelerator: Accelerator, tracker: Optional[Tracker] = None):
        super().__init__(cfg, accelerator, tracker)

    def generate_rollin(self, obs_tensor):
        gen_tokens = obs_tensor['reference_encoded_pt'][:, -self.max_gen_len:]
        seq_lens = gen_tokens.not_equal(self.tokenizer.pad_token_id).sum(axis=1).cpu()
        return gen_tokens, seq_lens
