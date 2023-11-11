import gc
from typing import Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from tril.algorithms.ppo import PPO
from tril.logging import Tracker
from tril.utils.policy import ModelType


class PPO_PP(PPO):
    def __init__(
        self,
        cfg,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        super().__init__(cfg, accelerator, tracker)

        self.rng = np.random.default_rng(seed=self.seed + 123)
        self.rollin_policy = "guide"
        self.rollout_policy = "policy"
        self.value_fn = "policy"
        self.accelerator.wait_for_everyone()

    def generate_rollin(self, obs_tensor):
        # Generating Rollins from Init State
        gen_output = self.accelerator.unwrap_model(self.agent.policy).generate(
            input_ids=obs_tensor["prompt_or_input_encoded_pt"],
            attention_mask=obs_tensor["prompt_or_input_attention_mask_pt"],
            accelerator=self.accelerator,
            actor_fn=self.rollin_policy,
            rng=self.rng,
        )

        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]
        if seq_length < self.max_gen_len:
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"
            padded_out = self.tokenizer.pad(
                {"input_ids": gen_tokens},
                padding="max_length",
                max_length=self.max_gen_len,
            )
            self.tokenizer.padding_side = prev_padding_side
            gen_tokens = padded_out["input_ids"].to(self.accelerator.device)

        gen_tokens = gen_tokens.cpu()
        seq_lens = gen_tokens.not_equal(self.tokenizer.pad_token_id).sum(axis=1).numpy()
        if not isinstance(gen_output, dict):
            delattr(gen_output, "sequences")
            delattr(gen_output, "scores")
        del gen_output
        gc.collect()
        torch.cuda.empty_cache()
        return gen_tokens, seq_lens

    def generate_batch(
        self,
        obs_tensor: Dict[str, torch.Tensor],
        rollin_actions: Optional[torch.Tensor] = None,
        rollin_seq_lens: Optional[List[int]] = None,
    ):
        # Note Rollin Mask is used to determine which segment of the trajectory to update # noqa
        # Given a full mixed sequence length of 5, if we have rollin mask [0, 0, 1, 1, 1], # noqa
        # we will only do our RL update w.r.t to the last 3 tokens.

        gen_output, rollin_mask = self.accelerator.unwrap_model(
            self.agent.policy
        ).generate(
            input_ids=obs_tensor["prompt_or_input_encoded_pt"],
            attention_mask=obs_tensor["prompt_or_input_attention_mask_pt"],
            accelerator=self.accelerator,
            actor_fn=self.rollout_policy,
            rollin_actions=rollin_actions,
            rollin_seq_lens=rollin_seq_lens,
            rng=self.rng,
            return_mask=True,
        )
        rollin_lengths = rollin_mask.sum(axis=1)

        # ===== Taken from Policy.generate ====
        seq_length = len(gen_output["scores"])
        all_tokens = gen_output["sequences"]

        if (
            self.accelerator.unwrap_model(self.agent.policy).model_type
            == ModelType.SEQ2SEQ
        ):
            # Gen output is decoder only => seq2seq we don't get prompt
            # Also Seq2seq prepends "start generation" token
            all_tokens = torch.cat(
                [obs_tensor["prompt_or_input_encoded_pt"], all_tokens[:, -seq_length:]],
                dim=1,
            )

        # Pad
        if seq_length < self.max_gen_len:
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"
            padded_out = self.tokenizer.pad(
                {"input_ids": all_tokens},
                padding="max_length",
                max_length=self.max_prompt_len + self.max_gen_len,
            )
            self.tokenizer.padding_side = prev_padding_side
            all_tokens = padded_out["input_ids"].to(self.accelerator.device)

        # Reward Computation
        terminal_rewards = self.agent.compute_reward(
            all_tokens=all_tokens,
            obs_tensor=obs_tensor,
            reference_map=self.reference_map,
        )

        # Everything is shape (Batch size, gen length)
        with torch.no_grad():
            obs = all_tokens[:, :-1]
            act = all_tokens[:, -self.max_gen_len :]
            policy_out = self.agent.policy.forward_actor(
                self.accelerator, obs, actions=act, actor_fn="policy"
            )
            ref_out = self.agent.policy.forward_actor(
                self.accelerator, obs, actions=act, actor_fn="ref"
            )
            value_out = self.agent.policy.forward_critic(
                self.accelerator, obs, value_fn=self.value_fn
            )

            # Grab outputs
            log_probs, entropies = policy_out.log_probs.cpu(), policy_out.entropy.cpu()
            ref_log_probs = ref_out.log_probs.cpu()
            values = value_out.values.cpu()

        all_tokens = all_tokens.cpu()
        masks = (
            all_tokens[:, -self.max_gen_len :]
            .not_equal(self.tokenizer.pad_token_id)
            .long()
        )
        seq_lens = masks[:, -self.max_gen_len :].sum(axis=1)

        # TODO: clean up
        rewards = torch.zeros_like(masks).float()  # NOT sure if needs to be float
        for i in range(rewards.shape[0]):
            rewards[i][seq_lens[i] - 1] = terminal_rewards[i]

        seq_lens = seq_lens - rollin_lengths

        # TODO: Create generic clean up
        delattr(gen_output, "sequences")
        delattr(gen_output, "scores")
        del gen_output
        gc.collect()
        torch.cuda.empty_cache()

        # KL Penalty
        kl_div = log_probs - ref_log_probs  # (B, gen_len)
        kl_rewards = -1 * self.kl_controller.kl_coeff * kl_div
        total_rewards = rewards.reshape(*kl_div.shape) + kl_rewards

        out = {
            "observation": all_tokens,
            "log_prob": log_probs,
            "value": values,
            "entropy": entropies,
            "ref_log_prob": ref_log_probs,
            "kl_div": kl_div,
            "kl_rewards": kl_rewards,
            "rewards": rewards,
            "episode_lengths": seq_lens,
            "total_rewards": total_rewards,
            "masks": masks,
            "rollin_masks": rollin_mask,
        }
        torch.cuda.empty_cache()
        return out

    def collect_rollouts(self):
        # Reset Buffer
        self.buffer.reset()

        # Set to inference mode
        self.accelerator.unwrap_model(self.agent.policy).train(False)

        # Setup
        if self.verbose > 0:
            self.accelerator.print(
                f"CURRENT AlG: {self.agent.policy.curr_alg_type} | Value Fn: {self.value_fn} | Rollin/Rollout: {self.rollin_policy}/{self.rollout_policy}"  # noqa
            )

        # Collect Samples
        n_sampling_steps = self.buffer.total_num_traj // self.buffer.num_traj_per_sample
        for _ in tqdm(
            range(n_sampling_steps),
            desc="Sampling",
            disable=not self.accelerator.is_local_main_process,
        ):
            assert not self.buffer.is_full()
            # start parallel episodes
            current_obs = next(self.prompt_sampler)

            # Get Reference
            target_ids = current_obs["reference_encoded_pt"]
            target_masks = current_obs["reference_attention_mask_pt"][
                :, -self.max_gen_len :
            ]

            obs_tensor = {
                k: v.to(self.accelerator.device) for k, v in current_obs.items()
            }

            # Collect Rollins
            rollin_actions, rollin_seq_lens = self.generate_rollin(
                obs_tensor=obs_tensor,
            )
            # Collect Rollouts
            batch = self.generate_batch(
                obs_tensor=obs_tensor,
                rollin_actions=rollin_actions,
                rollin_seq_lens=rollin_seq_lens,
            )

            # Add to Buffer
            self.buffer.batch_add(
                batch["observation"],
                batch["value"],
                batch["log_prob"],
                batch["rewards"],
                batch["masks"],
                target_ids,
                target_masks,
                batch["rollin_masks"],
            )

            # Log
            self.metric_tracker.add(
                batch, rollin_lengths=batch["rollin_masks"].sum(dim=-1)
            )

        # Gather Buffer
        self.buffer.gather_buffer(self.accelerator)
