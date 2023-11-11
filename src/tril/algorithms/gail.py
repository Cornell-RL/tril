import math
from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from tril.algorithms.ppo import PPO
from tril.buffers.prompt_buffer import infinite_dataloader
from tril.logging import Tracker


class GAIL(PPO):
    def __init__(
        self,
        cfg,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        super().__init__(cfg, accelerator, tracker)

        # Setting critic
        self.value_fn = "policy"

    def _prepare_deepspeed(self):
        self.accelerator.dispatch_batches = True
        self.buffer_dataloader = self.accelerator.prepare_data_loader(
            self.buffer_dataloader, device_placement=True
        )
        self.accelerator.dispatch_batches = False

        # Prepare
        assert self.agent.reward.is_trainable
        self.optimizer, self.reward_optimizer = self.agent.setup_optimizer()
        self.scheduler = self.agent.create_scheduler(
            self.optimizer, scheduler_args=self.cfg.get("scheduler", None)
        )
        (
            self.agent,
            self.optimizer,
            self.reward_optimizer,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.agent,
            self.optimizer,
            self.reward_optimizer,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
            self.scheduler,
        )

        self.prompt_sampler = infinite_dataloader(self.prompt_loader)

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
        assert self.agent.reward.is_trainable
        self.optimizer, self.reward_optimizer = self.agent.setup_optimizer()
        self.scheduler = self.agent.create_scheduler(
            self.optimizer, scheduler_args=self.cfg.get("scheduler", None)
        )
        (
            self.optimizer,
            self.reward_optimizer,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.optimizer,
            self.reward_optimizer,
            self.dataloaders["val"],
            self.dataloaders["test"],
            self.prompt_loader,
            self.scheduler,
        )

        self.prompt_sampler = infinite_dataloader(self.prompt_loader)

    def update_buffer(self):
        # Update Discriminator
        self.discriminator_step()

        # Loop over self.buffer.observations => score => self.buffer.rewards
        self.agent.eval()
        eval_batch_size = 32
        n_batches = math.ceil(self.buffer.observations.shape[0] / eval_batch_size)
        all_scores = []
        for i in range(n_batches):
            end = min((i + 1) * eval_batch_size, self.buffer.observations.shape[0])
            batch_obs = self.buffer.observations[i * eval_batch_size : end].to(
                self.accelerator.device
            )
            with torch.no_grad():
                scores = self.score(chosen_tokens=batch_obs, expert_tokens=None)
            all_scores.append(scores.cpu())

        # Populate terminal rewards in proper index
        terminal_rewards = torch.cat(all_scores, dim=0)
        seq_lens = self.buffer.masks.sum(axis=-1)
        self.buffer.rewards = torch.zeros(
            (self.trajectories_per_update, self.max_gen_len), dtype=torch.float32
        )
        for reward, length in zip(terminal_rewards, seq_lens):
            self.buffer.rewards[:, int(length - 1)] = reward

        # Update with updated buffer
        super().update_buffer()

    def score(self, *args, **kwargs):
        chosen_tokens = kwargs["chosen_tokens"]
        expert_tokens = kwargs["expert_tokens"]

        prompt_tokens = chosen_tokens[:, : self.max_prompt_len]
        gen_tokens = chosen_tokens[:, -self.max_gen_len :]

        if expert_tokens is not None:
            expert_prompt_tokens = expert_tokens[:, : self.max_prompt_len]
            expert_gen_tokens = expert_tokens[:, -self.max_gen_len :]
            prompt_tokens = torch.cat([prompt_tokens, expert_prompt_tokens], dim=0)
            gen_tokens = torch.cat([gen_tokens, expert_gen_tokens], dim=0)

        if self.agent.lora_cfg is not None:
            all_tokens = torch.cat([prompt_tokens, gen_tokens], dim=1)
            terminal_rewards = self.agent.forward(
                self.accelerator,
                self.tokenizer,
                all_tokens,
                forward_reward_only=True,
            )
        else:
            prompts = self.tokenizer.batch_decode(
                prompt_tokens.int(), skip_special_tokens=True
            )
            gens = self.tokenizer.batch_decode(
                gen_tokens.int(), skip_special_tokens=True
            )
            # refs = [reference_map[p] for p in prompts]
            refs = ["hi" for _ in range(len(prompts))]

            assert len(prompts) == len(gens) == len(refs)

            terminal_rewards = self.agent.forward(
                self.accelerator, prompts, gens, refs, forward_reward_only=True
            )
        return terminal_rewards

    def discriminator_step(self):
        self.agent.train()
        for epoch in tqdm(
            range(self.alg_cfg.args.discrim_epochs),
            desc="Discriminator Train Epoch",
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
                        # NOTE: we could just grab it from rollout_data.target_ids

                        chosen_tokens = rollout_data.observations.to(
                            self.accelerator.device
                        )
                        expert_tokens = rollout_data.target_ids.to(
                            self.accelerator.device
                        )

                        # Forward Function
                        scores = self.score(
                            chosen_tokens=chosen_tokens, expert_tokens=expert_tokens
                        )
                        chosen_scores, rejected_scores = torch.chunk(scores, 2)

                        # Compute Loss
                        loss = self.loss(chosen_scores, rejected_scores)

                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            pbar.update(1)

                        self.reward_optimizer.step()
                        self.reward_optimizer.zero_grad()

    def loss(self, chosen_scores, rejected_scores, method="cross-entropy"):
        # cross entropy loss
        if method == "cross-entropy":
            chosen_labels = torch.zeros_like(chosen_scores)
            rejected_labels = torch.ones_like(rejected_scores)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.cat([chosen_scores, rejected_scores]),
                torch.cat([chosen_labels, rejected_labels]),
            )
            return loss
        else:
            raise NotImplementedError(f"Loss method {method} not implemented")
