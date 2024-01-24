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

    #def collect_rollouts(self):
    #    # Reset Buffer
    #    self.buffer.reset()

    #    # Set to inference mode
    #    self.accelerator.unwrap_model(self.agent.policy).train(False)

    #    # Setup
    #    if self.verbose > 0:
    #        self.accelerator.print(
    #            f"CURRENT AlG: {self.agent.policy.curr_alg_type} | Value Fn: {self.value_fn} | Rollin/Rollout: {self.rollin_policy}/{self.rollout_policy}"  # noqa
    #        )

    #    # Collect Samples
    #    n_sampling_steps = self.buffer.total_num_traj // self.buffer.num_traj_per_sample
    #    for _ in tqdm(
    #        range(n_sampling_steps),
    #        desc="Sampling",
    #        disable=not self.accelerator.is_local_main_process,
    #    ):
    #        assert not self.buffer.is_full()
    #        # start parallel episodes
    #        current_obs = next(self.prompt_sampler)

    #        # Get Reference
    #        target_ids = current_obs["reference_encoded_pt"]
    #        target_masks = current_obs["reference_attention_mask_pt"][
    #            :, -self.max_gen_len :
    #        ]

    #        obs_tensor = {
    #            k: v.to(self.accelerator.device) for k, v in current_obs.items()
    #        }

    #        # Collect Rollins
    #        rollin_actions, rollin_seq_lens = self.generate_rollin(
    #            obs_tensor=obs_tensor,
    #        )
    #        # Collect Rollouts
    #        batch = self.generate_batch(
    #            obs_tensor=obs_tensor,
    #            rollin_actions=rollin_actions,
    #            rollin_seq_lens=rollin_seq_lens,
    #            anneal_beta=self.iteration >=40
    #        )

    #        # Add to Buffer
    #        self.buffer.batch_add(
    #            batch["observation"],
    #            batch["value"],
    #            batch["log_prob"],
    #            batch["rewards"],
    #            batch["masks"],
    #            target_ids,
    #            target_masks,
    #            batch["rollin_masks"],
    #        )

    #        # Log
    #        self.metric_tracker.add(
    #            batch, rollin_lengths=batch["rollin_masks"].sum(dim=-1)
    #        )

    #    # Gather Buffer
    #    self.buffer.gather_buffer(self.accelerator)
