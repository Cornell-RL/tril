from functools import partial
from typing import Optional

from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from tril.agent import Agent
from tril.base_algorithm import BaseAlgorithm
from tril.buffers.offline_buffer import create_dataloader
from tril.logging import LoggingTrainingMetrics, Tracker
from tril.utils.builders import build_metrics, build_task, build_tokenizer
from tril.utils.evaluation import evaluate_on_samples
from tril.utils.helpers import fsdp_prepare, preprocess_spice
from tril.utils.supervised import (
    get_datasets_for_causal,
    get_datasets_for_seq2seq,
    tokenize_causal,
    tokenize_seq2seq,
)


class BaseSupervised(BaseAlgorithm):
    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        self.metric_tracker = LoggingTrainingMetrics(prefix="train")

        self.tokenizer_cfg = cfg.alg.tokenizer
        self.task_cfg = cfg.task
        self.alg_cfg = cfg.alg
        self.lora_cfg = cfg.alg.get("lora", None)
        self.sampling_cfg = cfg.sampling

        self.max_prompt_len = self.sampling_cfg.max_prompt_len
        self.max_gen_len = self.sampling_cfg.max_gen_len

        self.model_type = self.alg_cfg.policy.args.model_type

        super().__init__(cfg=cfg, accelerator=accelerator, tracker=tracker)

    def _setup(self):
        self.tokenizer = build_tokenizer(self.tokenizer_cfg)
        self.metrics = build_metrics(self.cfg.get("eval_metrics", []), self.accelerator)

        self.samples_by_split = build_task(self.task_cfg)

        for metric in self.metrics:
            if hasattr(metric, "_spice_metric"):
                preprocess_spice(metric, self.samples_by_split, self.accelerator)
                break

        self.gen_kwargs = self.alg_cfg.policy.args.gen_kwargs
        self.agent = Agent(self.cfg, self.accelerator, self.tokenizer)

        # Create DataLoaders
        self._setup_dataloaders()

        # Prepare for Accelerate
        self._prepare_accelerate()

    def _prepare_fsdp(self):
        self.agent = self.accelerator.prepare(self.agent)

        # NOTE: for fsdp it is important to create optimizer AFTER preparing model
        self.optimizer = self.agent.setup_optimizer()
        (
            self.optimizer,
            self.train_dataloader,
            self.dataloaders["val"],
            self.dataloaders["test"],
        ) = self.accelerator.prepare(
            self.optimizer,
            self.dataloaders["train"],
            self.dataloaders["val"],
            self.dataloaders["test"],
        )

    def _prepare_deepspeed(self):
        self.optimizer = self.agent.setup_optimizer()
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.dataloaders["val"],
            self.dataloaders["test"],
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.dataloaders["train"],
            self.dataloaders["val"],
            self.dataloaders["test"],
        )

    def _setup_dataloaders(self):
        train_dataset = (
            get_datasets_for_causal(self.samples_by_split["train"])
            if self.model_type == "causal"
            else get_datasets_for_seq2seq(self.samples_by_split["train"])
        )
        preprocess_fn = (
            tokenize_causal if self.model_type == "causal" else tokenize_seq2seq
        )
        preprocess_fn = partial(preprocess_fn, tokenizer=self.tokenizer)
        with self.accelerator.local_main_process_first():
            self.tokenized_dataset = train_dataset.map(
                preprocess_fn, batched=True, remove_columns=train_dataset.column_names
            )

        self.data_collator = (
            DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            if self.model_type == "causal"
            else DataCollatorForSeq2Seq(self.tokenizer, self.model)
        )
        train_dataloader = DataLoader(
            self.tokenized_dataset,
            batch_size=self.alg_cfg.args.batch_size_per_process,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        # Setup Evaluation
        self.eval_gen_kwargs = self.sampling_cfg.get(
            "evaluation_generation_kwargs", None
        )

        self.dataloaders = {
            "train": train_dataloader,
            "val": create_dataloader(
                self.samples_by_split["val"], self.eval_batch_size
            ),
            "test": create_dataloader(
                self.samples_by_split["test"], self.eval_batch_size
            ),
        }

    def eval_step(self, epoch: int):
        if self.dist_type == DistributedType.FSDP:
            fsdp_prepare(
                self.agent,
                self.tokenizer,
                self.accelerator,
                self.max_prompt_len + self.max_gen_len,
                supervised=True,
            )
        for split in self.eval_splits:
            evaluate_on_samples(
                policy=self.agent.policy,
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
        self.train_step()
