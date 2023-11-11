import time
from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from tril.algorithms.base_supervised import BaseSupervised
from tril.logging import Tracker


class BC(BaseSupervised):
    def __init__(
        self,
        cfg,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        super().__init__(cfg, accelerator, tracker)

    def compute_loss(self, inputs):
        """
        Override for custom losses
        """
        output = self.agent(**inputs)
        return output[0].mean()

    def train_step(self):
        # Wait for all Initialization
        self.accelerator.wait_for_everyone()
        self.start_time = time.time_ns()
        pbar = tqdm(
            range(self.alg_cfg.args.n_epochs * len(self.train_dataloader)),
            disable=not self.accelerator.is_main_process,
        )
        for epoch_idx, epoch in enumerate(range(self.alg_cfg.args.n_epochs)):
            step = 0
            for batch in self.train_dataloader:
                progress_so_far = epoch_idx + (step / len(self.train_dataloader))

                # Train
                self.agent.train(True)
                with self.accelerator.accumulate(self.agent):
                    loss = self.compute_loss(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Log
                    self.metric_tracker.add("loss", loss.item())

                    if self.accelerator.sync_gradients:
                        # TODO: if we want gradient clipping add here
                        pbar.update(1)
                        pbar.set_description(
                            f"Epoch: {progress_so_far:.2f} | Training loss: {loss.item():.4f}"  # noqa
                        )

                # Eval
                if step % self.eval_every == 0:
                    self.agent.eval()
                    self.eval_step(progress_so_far)

                    # Log
                    self.accelerator.wait_for_everyone()
                    training_info = self.metric_tracker.metrics_for_gather(
                        self.accelerator
                    )
                    aggregated_training_info = self.accelerator.gather(training_info)
                    aggregated_training_info = {
                        key: torch.mean(value).item()
                        for key, value in aggregated_training_info.items()
                    }
                    aggregated_training_info["train/epoch"] = progress_so_far
                    self.tracker.log_training_infos(aggregated_training_info)
                step += 1
