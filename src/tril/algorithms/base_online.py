import gc
import sys
import time
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig
from tqdm import tqdm

from tril.agent import Agent
from tril.base_algorithm import BaseAlgorithm
from tril.base_reward import RewardType
from tril.buffers.offline_buffer import create_dataloader
from tril.buffers.online_buffer import OnlineBuffer
from tril.buffers.prompt_buffer import create_prompt_dataloader, infinite_dataloader
from tril.logging import LoggingSamplingMetrics, LoggingTrainingMetrics, Tracker
from tril.utils.builders import build_metrics, build_task, build_tokenizer
from tril.utils.evaluation import evaluate_on_samples
from tril.utils.helpers import (
    TorchTracemalloc,
    fsdp_prepare,
    fsdp_reward_prepare,
    get_schedule_fn,
    preprocess_spice,
    print_memory,
)
from tril.utils.kl_controller import KLRegistry
from tril.utils.policy import ModelType


class BaseOnPolicyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        # Logger
        keys_to_log = {
            "total_rewards": "trajectory",
            "kl_rewards": "trajectory",
            "rewards": "trajectory",
            "kl_div": "trajectory",
            "value": "sample",
            "entropy": "sample",
        }
        self.metric_tracker = LoggingSamplingMetrics(keys_to_log)
        self.alg_metric_tracker = LoggingTrainingMetrics()

        # Configs
        self.tokenizer_cfg = cfg.alg.tokenizer
        self.task_cfg = cfg.task
        self.sampling_cfg = cfg.sampling
        self.policy_cfg = cfg.alg.policy
        self.lora_cfg = cfg.alg.get("lora", None)

        super().__init__(cfg=cfg, accelerator=accelerator, tracker=tracker)

    def _setup(self):
        # Check config values
        sampling_check = self.trajectories_per_update % (
            self.sampling_cfg.batch_size_per_process * self.num_processes
        )
        if sampling_check != 0:
            raise ValueError(
                "`trajectories_per_update` needs to be divisible by `batch_size_per_process` * `num_processes` for proper distributed gpu training. Please edit these values"  # noqa
            )
        batch_check = self.batch_size % (
            self.grad_accumulation_steps * self.num_processes
        )
        if batch_check != 0:
            raise ValueError(
                "Set `batch_size` must be achievable with set `grad_accumululation` and `num_processes`. Please edit these values"  # noqa
            )
        minibatch_check = self.trajectories_per_update % self.batch_size
        if minibatch_check != 0:
            raise ValueError(
                "`trajectories_per_update` needs to be divisible by `batch_size` for proper training. Please edit these values"  # noqa
            )

        # Build Components
        self.tokenizer = build_tokenizer(self.tokenizer_cfg)
        self.agent = Agent(
            self.cfg,
            self.accelerator,
            self.tokenizer,
        )

        self.metrics = build_metrics(self.cfg.get("eval_metrics", []), self.accelerator)
        self.samples_by_split = build_task(self.task_cfg)

        if hasattr(self.agent.reward, "_spice_metric"):
            assert self.agent.reward is not None

            if self.accelerator.is_main_process:
                preprocess_spice(
                    self.agent.reward._spice_metric,
                    self.samples_by_split,
                    self.accelerator,
                )
            self.accelerator.wait_for_everyone()

        # KL Controller
        kl_cfg = self.alg_cfg.kl_div
        kl_controller_cls = KLRegistry.get(kl_cfg.kl_type)
        self.kl_controller = kl_controller_cls(
            kl_cfg.coeff,
            kl_cfg.kl_lr,
            kl_cfg.get("target_kl", None),
        )

        # Max # of Tokens processed in entire training
        self.total_timesteps = (
            self.trajectories_per_update * self.max_gen_len * self.n_iters
        )

        self.tracker.log_info(f"Total steps for algorithms: {self.total_timesteps}")
        self.current_progress_remaining = 1.0

        # Initialize schedules for policy/value clipping
        # TODO: add in preprocess schedules here
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # Setup Datasources
        self._setup_dataloaders()

        # prepare for accelerate
        self._prepare_accelerate()

    def _setup_dataloaders(self):
        # Prompt Sampling
        self.prompt_loader = create_prompt_dataloader(
            self.sampling_cfg.batch_size_per_process,
            self.samples_by_split["train"],
            self.tokenizer,
            self.sampling_cfg.max_prompt_len,
            self.sampling_cfg.max_gen_len,
            prompt_truncation_side=self.sampling_cfg.prompt_truncation_side,
            context_truncation_side=self.sampling_cfg.context_truncation_side,
            prompt_padding_side=self.sampling_cfg.prompt_padding_side,
            context_padding_side=self.sampling_cfg.context_padding_side,
        )
        # Processed Prompts[str] -> References[List[str]]
        self.reference_map = self.prompt_loader.dataset.reference_map

        # Sample Buffer per Process
        self.buffer = OnlineBuffer(
            self.accelerator,
            trajectories_in_buffer=self.trajectories_per_update // self.num_processes,
            trajectories_per_sample=self.sampling_cfg.batch_size_per_process,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            max_prompt_len=self.max_prompt_len,
            max_gen_len=self.max_gen_len,
        )

        if (
            self.alg_cfg["args"]["batch_size"]
            % (self.num_processes * self.accelerator.gradient_accumulation_steps)
            != 0
        ):
            # NOTE: does not NEED to... but for now leaving this
            raise ValueError(
                f"Alg Batch size needs to divide by the number of processes ({self.num_processes}) and grad accumulation steps ({self.accelerator.gradient_accumulation_steps})"  # noqa
            )
        per_process_batch = int(
            self.alg_cfg["args"]["batch_size"]
            / (self.num_processes * self.accelerator.gradient_accumulation_steps)
        )
        self.buffer_dataloader = self.buffer.create_dataloader(
            per_process_batch, shuffle=True
        )

        # Setup Evaluation
        self.eval_gen_kwargs = self.sampling_cfg.eval_generation_kwargs

        self.dataloaders = {
            "val": create_dataloader(
                self.samples_by_split["val"], self.eval_batch_size
            ),
            "test": create_dataloader(
                self.samples_by_split["test"], self.eval_batch_size
            ),
        }

    def _prepare_fsdp(self):
        self.accelerator.dispatch_batches = True
        self.buffer_dataloader = self.accelerator.prepare_data_loader(
            self.buffer_dataloader, device_placement=True
        )
        self.accelerator.dispatch_batches = False

        # Prepare
        if self.lora_cfg:
            raise ValueError(
                "Using FSDP with lora is not recommended so we don't support. Use Deepspeed instead"  # noqa
            )

        self.agent = self.accelerator.prepare(self.agent)

        # prepare optimizer(s) and dataloaders
        assert not self.agent.reward.is_trainable
        self.optimizer = self.agent.setup_optimizer()
        self.scheduler = self.agent.create_scheduler(
            self.optimizer, scheduler_args=self.alg_cfg.get("scheduler", None)
        )
        (
            self.optimizer,
            self.scheduler,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
        ) = self.accelerator.prepare(
            self.optimizer,
            self.scheduler,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
        )

        self.prompt_sampler = infinite_dataloader(self.prompt_loader)

    def _prepare_deepspeed(self):
        self.accelerator.dispatch_batches = True
        self.buffer_dataloader = self.accelerator.prepare_data_loader(
            self.buffer_dataloader, device_placement=True
        )
        self.accelerator.dispatch_batches = False

        # Prepare
        assert not self.agent.reward.is_trainable
        self.optimizer = self.agent.setup_optimizer()
        self.scheduler = self.agent.create_scheduler(
            self.optimizer, scheduler_args=self.alg_cfg.get("scheduler", None)
        )
        (
            self.agent,
            self.optimizer,
            self.scheduler,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
        ) = self.accelerator.prepare(
            self.agent,
            self.optimizer,
            self.scheduler,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
        )

        self.prompt_sampler = infinite_dataloader(self.prompt_loader)

    def generate_batch(
        self, obs_tensor: Dict[str, torch.Tensor], gen_kwargs: Dict[str, Any] = None
    ):
        gen_output = self.accelerator.unwrap_model(self.agent.policy).generate(
            input_ids=obs_tensor["prompt_or_input_encoded_pt"],
            attention_mask=obs_tensor["prompt_or_input_attention_mask_pt"],
            accelerator=self.accelerator,
            gen_kwargs=gen_kwargs,
        )
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
                self.accelerator, obs, actions=act, model_fn="policy"
            )
            ref_out = self.agent.policy.forward_actor(
                self.accelerator, obs, actions=act, model_fn="ref"
            )
            value_out = self.agent.policy.forward_critic(self.accelerator, obs)

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
        }
        torch.cuda.empty_cache()
        return out

    def collect_rollouts(self):
        # Reset Buffer
        self.buffer.reset()

        # set to inference
        self.accelerator.unwrap_model(self.agent).train(False)

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

            # Collect Samples
            batch = self.generate_batch(obs_tensor)

            # Add to Buffer
            self.buffer.batch_add(
                batch["observation"],
                batch["value"],
                batch["log_prob"],
                batch["total_rewards"],
                batch["masks"],
                target_ids,
                target_masks,
            )

            # Log
            self.metric_tracker.add(batch)

        # Gather Buffer
        self.buffer.gather_buffer(self.accelerator)

    def update_buffer(self):
        # Advantage Computation
        self.buffer.compute_returns_and_advantage()

        # Gather Metrics
        self.accelerator.wait_for_everyone()
        metrics_for_gather = self.metric_tracker.metrics_for_gather(self.accelerator)
        gathered_metrics = self.accelerator.gather_for_metrics(metrics_for_gather)
        for k, v in gathered_metrics.items():
            gathered_metrics[k] = torch.mean(v).item()

        # Controller update
        if self.accelerator.is_main_process:
            gathered_metrics["rollout_buffer/kl_coef"] = self.kl_controller.kl_coeff
            self.kl_controller.step(gathered_metrics["rollout_buffer/kl_div"])

        if self.tracker is not None:
            self.tracker.log_rollout_infos(gathered_metrics)

    def eval_step(self, epoch: int):
        if self.dist_type == DistributedType.FSDP:
            fsdp_prepare(
                self.agent,
                self.tokenizer,
                self.accelerator,
                self.max_prompt_len + self.max_gen_len,
            )
            if (
                not self.agent.reward.is_trainable
                and self.agent.reward._dist_type == RewardType.DIST
            ):
                fsdp_reward_prepare(self.agent.reward, self.accelerator)
        for split in self.eval_splits:
            evaluate_on_samples(
                agent=self.agent,
                tokenizer=self.tokenizer,
                dataloader=self.dataloaders[split],
                max_prompt_length=self.max_prompt_len,
                metrics=self.metrics,
                epoch=epoch,
                split_name=split,
                accelerator=self.accelerator,
                tracker=self.tracker,
                gen_kwargs=self.eval_gen_kwargs,
            )

    def learn(self):
        # Wait for all Initialization
        self.accelerator.wait_for_everyone()

        if self.eval_zero_shot:
            self.eval_step(epoch=0)

        self.iteration, num_timesteps = 0, 0
        self.start_time = time.time_ns()
        while num_timesteps < self.total_timesteps:
            # ========= Logging ==========
            if self.accelerator.is_main_process:
                # Iteration FPS
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
                )
                fps = int(num_timesteps / time_elapsed)

                # Log
                learn_info = {
                    "time/fps": fps,
                    "time/total_timesteps": num_timesteps,
                    "time/iterations": self.iteration,
                }
                self.tracker.log_training_infos(learn_info)

            if self.accelerator.distributed_type == DistributedType.FSDP:
                fsdp_prepare(
                    self.agent,
                    self.tokenizer,
                    self.accelerator,
                    self.max_prompt_len + self.max_gen_len,
                )
                if (
                    not self.agent.reward.is_trainable
                    and self.agent.reward._dist_type == RewardType.DIST
                ):
                    fsdp_reward_prepare(self.agent.reward, self.accelerator)

            # ========= Sampling =========
            with TorchTracemalloc() as tracemalloc:
                self.collect_rollouts()

                # Gather and Update Buffer Values
                self.update_buffer()

            if self.verbose > 0:
                print_memory(self.accelerator, tracemalloc, "sampling")

            # =========== Train ===========
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            with TorchTracemalloc() as tracemalloc:
                self.train_step()
            if self.verbose > 0:
                print_memory(self.accelerator, tracemalloc, "train")

            self.iteration += 1

            # Number of Trajectories * Max Gen Tokens * Num Processes
            num_timesteps += len(self.buffer) * self.max_gen_len
            self.current_progress_remaining = 1.0 - (
                num_timesteps / self.total_timesteps
            )

            # ========= Evaluation ========
            if self.iteration % self.eval_every == 0:
                with TorchTracemalloc() as tracemalloc:
                    self.eval_step(epoch=self.iteration)
                if self.save_checkpoints:
                    # TODO: saving reward weights as well
                    self.tracker.save_auto_model(
                        self.agent.policy, self.accelerator, self.iteration
                    )
                if self.verbose > 0:
                    print_memory(self.accelerator, tracemalloc, "eval")
