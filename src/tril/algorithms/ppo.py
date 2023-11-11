from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from tril.algorithms.base_online import BaseOnPolicyAlgorithm
from tril.logging import Tracker
from tril.utils.helpers import explained_variance


class PPO(BaseOnPolicyAlgorithm):
    def __init__(
        self,
        cfg,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        super().__init__(cfg, accelerator, tracker)

        # Setting critic
        self.value_fn = "policy"

    def train_step(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.accelerator.unwrap_model(self.agent).train(True)

        # Compute current clip range
        clip_range = self.clip_range(self.current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self.current_progress_remaining)

        continue_training = True

        for epoch in tqdm(
            range(self.n_epochs),
            desc="PPO Train Epoch",
            disable=not self.accelerator.is_local_main_process,
        ):
            # Do a complete pass on the rollout buffer
            n_minibatches = self.trajectories_per_update // self.batch_size
            with tqdm(
                total=n_minibatches,
                desc="MiniBatches",
                disable=not self.accelerator.is_local_main_process,
                leave=False,
            ) as pbar:
                for batch_ix, rollout_data in enumerate(self.buffer_dataloader):
                    with self.accelerator.accumulate():
                        # Masking Termination + Rollins
                        masks = (rollout_data.masks * rollout_data.rollin_masks).to(
                            self.accelerator.device
                        )
                        n_samples = torch.sum(masks)

                        observations = rollout_data.observations[:, :-1].to(
                            self.accelerator.device
                        )
                        actions = rollout_data.observations[:, -self.max_gen_len :].to(
                            self.accelerator.device
                        )

                        evaluation_output = self.agent.forward(
                            accelerator=self.accelerator,
                            obs=observations,
                            actions=actions,
                            value_fn=self.value_fn,
                            forward_policy_only=True,
                        )
                        # pdb.set_trace()

                        values, log_prob, entropy = (
                            evaluation_output.values,
                            evaluation_output.log_prob,
                            evaluation_output.entropy,
                        )

                        # Forward KL Target Reference Loss
                        target_loss = torch.tensor(0.0)
                        if self.target_regularization:
                            n_target = torch.sum(rollout_data.target_masks)
                            target_out = self.agent.forward(
                                accelerator=self.accelerator,
                                obs=rollout_data.target_ids[:, :-1].to(
                                    self.accelerator.device
                                ),
                                actions=rollout_data.target_ids[
                                    :, -self.max_gen_len :
                                ].to(self.accelerator.device),
                                forward_actor=True,
                                forward_policy_only=True,
                            )
                            target_loss = (
                                -torch.sum(
                                    target_out.log_probs
                                    * rollout_data.target_masks.to(
                                        self.accelerator.device
                                    )
                                )
                                / n_target
                            )

                        # Normalize advantage
                        advantages = rollout_data.advantages.to(self.accelerator.device)

                        # ratio between old and new policy
                        log_ratio = (
                            log_prob
                            - rollout_data.old_log_prob.to(self.accelerator.device)
                        ) * masks
                        ratio = torch.exp(log_ratio)
                        with torch.no_grad():
                            approx_kl_div = torch.mean((ratio - 1) - log_ratio).cpu()

                        # clipped surrogate loss
                        policy_loss_1 = advantages * ratio
                        policy_loss_2 = advantages * torch.clamp(
                            ratio, 1 - clip_range, 1 + clip_range
                        )
                        policy_loss = (
                            -torch.sum(torch.min(policy_loss_1, policy_loss_2) * masks)
                            / n_samples
                        )
                        clip_fraction = (
                            torch.sum(
                                (torch.abs(ratio.detach() - 1) > clip_range).float()
                                * masks
                            )
                            / n_samples
                        )
                        # pdb.set_trace()

                        # Entropy loss favor exploration
                        if entropy is None:
                            # Approximate entropy when no analytical form
                            entropy_loss = -torch.sum(-log_prob * masks) / n_samples
                        else:
                            entropy_loss = -torch.sum(entropy * masks) / n_samples

                        # Value Loss
                        value_masks, n_value = masks, n_samples
                        if self.agent.policy.curr_alg_type == "aggrevate":
                            value_masks = (
                                ~rollout_data.rollin_masks.to(
                                    self.accelerator.device
                                ).bool()
                            ).float() * rollout_data.masks.to(self.accelerator.device)
                            n_value = torch.sum(value_masks)

                        returns = rollout_data.returns.to(self.accelerator.device)
                        val_delta = values - rollout_data.old_values.to(
                            self.accelerator.device
                        )
                        if self.clip_range_vf is not None:
                            clipped_values = torch.clamp(
                                values,
                                rollout_data.old_values - clip_range_vf,
                                rollout_data.old_values + clip_range_vf,
                            )
                            value_loss_1 = (values - returns) ** 2
                            value_loss_2 = (clipped_values - returns) ** 2
                            value_loss = (
                                0.5
                                * torch.sum(
                                    torch.max(value_loss_1, value_loss_2) * value_masks
                                )
                                / n_value
                            )
                            value_clip_fraction = (
                                torch.sum(
                                    (
                                        torch.abs(val_delta.detach()) > clip_range_vf
                                    ).float()
                                    * value_masks
                                )
                                / n_value
                            )
                        else:
                            value_loss = (
                                0.5
                                * torch.sum(value_masks * (values - returns) ** 2)
                                / n_value
                            )
                            value_clip_fraction = torch.tensor(0.0)

                        # Calculate Value Metrics
                        explained_var_in_batch = explained_variance(
                            values, rollout_data.returns
                        )
                        explained_var_out_batch = explained_variance(
                            rollout_data.old_values, rollout_data.returns
                        )

                        loss = (
                            policy_loss
                            + self.ent_coef * entropy_loss
                            + self.vf_coef * value_loss
                            + self.target_coef * target_loss
                        )

                        if (
                            self.target_kl is not None
                            and approx_kl_div > 1.5 * self.target_kl
                        ):
                            continue_training = False
                            if self.verbose >= 1:
                                print(
                                    f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"  # noqa
                                )
                            break

                        # Loss backward
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.agent.policy_params, self.max_grad_norm
                            )

                            # Update Progress Bar
                            pbar.update(1)

                        # Optimization Step
                        self.optimizer.step()

                        # Optimization step
                        self.optimizer.zero_grad()

                        # Log Metrics
                        self.alg_metric_tracker.add("target_loss", target_loss.item())
                        self.alg_metric_tracker.add("approx_kl", approx_kl_div.item())
                        self.alg_metric_tracker.add("ratio", ratio.mean().item())
                        self.alg_metric_tracker.add(
                            "policy_gradient_loss", policy_loss.item()
                        )
                        self.alg_metric_tracker.add(
                            "clip_fraction", clip_fraction.item()
                        )
                        self.alg_metric_tracker.add(
                            "val_clip_fraction", value_clip_fraction.item()
                        )
                        self.alg_metric_tracker.add("value_loss", value_loss.item())
                        self.alg_metric_tracker.add("entropy_loss", entropy_loss.item())
                        self.alg_metric_tracker.add(
                            "explained_var_in_batch", explained_var_in_batch.item()
                        )
                        self.alg_metric_tracker.add(
                            "explained_var_out_batch", explained_var_out_batch.item()
                        )
                        self.alg_metric_tracker.add("loss", loss.item())
                        if self.verbose >= 1:
                            self.alg_metric_tracker.add(
                                "value_delta", val_delta.mean().item()
                            )
                            self.alg_metric_tracker.add(
                                "advantages", advantages.mean().item()
                            )

                        pad_percent = 100 * (
                            1.0 - n_samples / torch.prod(torch.tensor(masks.shape))
                        )
                        self.alg_metric_tracker.add("pad_percent", pad_percent.item())

                        # Empty Cache for memory
                        torch.cuda.empty_cache()

            if not continue_training:
                break

        # Log
        self.accelerator.wait_for_everyone()
        training_info = self.alg_metric_tracker.metrics_for_gather(self.accelerator)
        aggregated_training_info = self.accelerator.gather(training_info)
        aggregated_training_info = {
            key: torch.mean(value).item()
            for key, value in aggregated_training_info.items()
        }
        aggregated_training_info["ppo/clip_range"] = clip_range
        aggregated_training_info["ppo/lr"] = self.scheduler.get_last_lr()[
            0
        ]  # Grab for single group
        if self.clip_range_vf is not None:
            aggregated_training_info["ppo/clip_range_vf"] = clip_range_vf

        # Track
        self.tracker.log_training_infos(aggregated_training_info)

        # Learning Rate Scheduler Step
        self.scheduler.step()
