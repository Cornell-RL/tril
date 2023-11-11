import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

import jsonlines
import pandas as pd
import torch
import wandb
from rich.logging import RichHandler


class Tracker:
    def __init__(
        self,
        base_path_to_store_results: str,
        run_config: Dict[str, Any],
        project_name: str,
        experiment_name: str,
        entity_name: str = None,
        wandb_log: bool = False,
        log_level: int = logging.INFO,
        is_main_process: bool = False,
    ):
        self._log_level = log_level
        self._run_path = base_path_to_store_results
        self._config = run_config
        self._experiment_name = experiment_name
        self._project_name = project_name
        self._entity_name = entity_name
        self._wandb_log = wandb_log
        self._is_main_process = is_main_process
        self._init()

    def _init(self):
        # store also the config into it
        config_path = os.path.join(self._run_path, "config.json")
        with open(config_path, "w") as fp:
            json.dump(self._config, fp)

        # init logger
        log_path = os.path.join(self._run_path, "log.txt")
        logging.basicConfig(
            level=self._log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path), RichHandler()],
        )

        # init wandb
        if self._wandb_log and self._is_main_process:
            self._wandb_run = wandb.init(
                entity=self._entity_name,
                project=self._project_name,
                name=self._experiment_name,
                config=self._config,
            )

            # set the wandb x axis
            # define our custom x axis metric
            self._wandb_run.define_metric("time/iterations")

            # set all other train/ metrics to use this step
            self._wandb_run.define_metric("ppo/*", step_metric="time/iterations")
            self._wandb_run.define_metric(
                "rollout_buffer/*", step_metric="time/iterations"
            )

            # TODO: clean up EVAL wandb logging
            self._wandb_run.define_metric(
                "val/diversity_metrics/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "val/fluency_metrics/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "val/semantic/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "val/lexical/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "test/diversity_metrics/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "test/fluency_metrics/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "test/semantic/*", step_metric="time/iterations"
            )
            self._wandb_run.define_metric(
                "test/lexical/*", step_metric="time/iterations"
            )

    def log_predictions(self, epoch: int, split_name: str, predictions: List[Dict]):
        if self._is_main_process:
            # log them per epoch in a separate file as they can get huge
            prediction_file_at_epoch = os.path.join(
                self._run_path, f"epoch_{epoch}_{split_name}_split_predictions.json"
            )
            with open(prediction_file_at_epoch, "w") as fp:
                json.dump(predictions, fp)

            # randomly display few predictions for logging
            # predictions_ = copy.deepcopy(predictions)
            # TODO: flag
            # random.shuffle(predictions_)
            # logging.info(f"Split {split_name} predictions")
            # for pred in predictions_[:10]:
            #    logging.info(pred)

            # for wandb logging, we create a table consisting of predictions
            # we can create one table per split per epoch
            if self._wandb_log and len(predictions) > 0:

                def to_df(predictions):
                    columns = predictions[0].keys()
                    data_by_column = defaultdict(list)
                    for item in predictions:
                        for column in columns:
                            data_by_column[column].append(item.get(column, ""))
                    data_df = pd.DataFrame(data_by_column)
                    return data_df

                predictions_as_df = to_df(predictions)
                predictions_table_at_epoch = wandb.Table(data=predictions_as_df)
                self._wandb_run.log(
                    {
                        f"{split_name}_predictions_at_epoch_{epoch}": predictions_table_at_epoch  # noqa
                    }
                )

    def log_metrics(self, epoch: int, split_name: str, metrics_dict: Dict[str, float]):
        if self._is_main_process:
            # for each split, one file
            metric_file_per_split = os.path.join(
                self._run_path, f"{split_name}_split_metrics.jsonl"
            )
            metrics_dict_ = {"epoch": epoch, "metrics": metrics_dict}
            with jsonlines.open(metric_file_per_split, "a") as writer:
                writer.write(metrics_dict_)

            # log to wandb
            if self._wandb_log:
                metric_dict_ = {
                    f"{split_name}/{metric_key}": value
                    for metric_key, value in metrics_dict.items()
                }
                # metric_dict_["epoch"] = epoch
                wandb.log(metric_dict_)

            # logger
            logging.info(f"{split_name} metrics: {metrics_dict_}")

    def log_memory_usage(self, memory_info: Dict[str, float]):
        if self._is_main_process:
            logging.info(f"Memory Info: {memory_info}")
            if self._wandb_log:
                wandb.log(memory_info)

    def log_rollout_infos(self, rollout_info: Dict[str, float]):
        if self._is_main_process:
            logging.info(f"Rollout Info: {rollout_info}")
            rollout_info_file = os.path.join(self._run_path, "rollout_info.jsonl")
            with jsonlines.open(rollout_info_file, mode="a") as writer:
                writer.write(rollout_info)

            # log to wandb
            if self._wandb_log:
                wandb.log(rollout_info)

    def log_training_infos(self, training_info: Dict[str, float]):
        if self._is_main_process:
            logging.info(f"Training Info: {training_info}")
            training_info_file = os.path.join(self._run_path, "training_info.jsonl")
            with jsonlines.open(training_info_file, mode="a") as writer:
                writer.write(training_info)

            # log to wandb
            if self._wandb_log:
                wandb.log(training_info)

    def done(self):
        if self._wandb_log:
            wandb.finish()

    # def save_auto_model(self, model: AutoModel):
    #    if self._is_main_process:
    #        model_path = os.path.join(self._run_path, "model")
    #        model.save_pretrained(model_path)
    # def save_auto_model(
    # self, model: AutoModel, accelerator, value_head=None, value=False):
    #    if self._is_main_process:
    #        if value:
    #            assert value_head is not None
    #            model_path = os.path.join(self._run_path, "value_model")
    #        else:
    #            model_path = os.path.join(self._run_path, "model")
    #        model.save_pretrained(
    #            model_path,
    #            is_main_process=accelerator.is_main_process,
    #            save_function=accelerator.save,
    #            state_dict=accelerator.get_state_dict(model),
    #        )
    #        if value:
    #            value_head_state = OrderedDict()
    #            weight = value_head.weight.clone().detach().cpu()
    #            bias = value_head.bias.clone().detach().cpu()
    #            value_head_state['weight'] = weight
    #            value_head_state['bias'] = bias
    #            with open(os.path.join(model_path, 'value_head.pt'), 'wb') as f:
    #                torch.save(value_head_state, f)
    #            del value_head_state
    def save_auto_model(self, policy, accelerator, iteration):
        if accelerator.is_main_process:
            save_path = os.path.join(self._run_path, f"model_{iteration}")
            unwrapped_model = accelerator.unwrap_model(policy._policy_model)
            unwrapped_model.save_pretrained(save_path, save_function=accelerator.save)

    @property
    def checkpoint_base_path(self):
        return os.path.join(self._run_path, "checkpoints")

    def log_info(self, msg: str):
        if self._is_main_process:
            logging.info(msg)


class LoggingSamplingMetrics:
    """Trajectory based logging. Useful when tracking metrics during batched sampling process."""  # noqa

    def __init__(self, keys_to_log, prefix="rollout_buffer"):
        self._metrics = defaultdict(list)
        self._reductions = keys_to_log
        self._prefix = prefix
        self._episode_lengths = []
        self._rollin_lengths = []

    def add(self, batch, rollin_lengths=None):
        for key, value in batch.items():
            if key in self._reductions.keys():
                self._metrics[key].append(value)
            if key == "episode_lengths":
                self._episode_lengths.append(batch[key])
        if rollin_lengths is not None:
            self._rollin_lengths.append(rollin_lengths)

    def reset(self):
        self._metrics = defaultdict(list)
        self._episode_lengths = []
        self._rollin_lengths = []

    def metrics_for_gather(self, accelerator, reset=True):
        metrics = {}
        episode_lengths = torch.cat(self._episode_lengths)
        rollin_lengths = (
            torch.zeros_like(episode_lengths)
            if len(self._rollin_lengths) == 0
            else torch.cat(self._rollin_lengths)
        )
        for k, v in self._metrics.items():
            reduction = self._reductions[k]
            batch_v = torch.cat(v, dim=0)
            result_key = "/".join([self._prefix, k])
            if reduction == "sample":
                metrics[result_key] = batch_v.flatten().to(accelerator.device)
            elif reduction == "trajectory":
                traj_v = []
                lengths = episode_lengths + rollin_lengths
                for ep, start, length in zip(batch_v, rollin_lengths, lengths):
                    traj_v.append(torch.sum(ep[start:length]))
                metrics[result_key] = torch.stack(traj_v).to(accelerator.device)
                # TODO: think of how not to hardcode
                if k == "kl_div":
                    metrics["/".join([self._prefix, "sqrt_kl"])] = torch.sqrt(
                        torch.abs(metrics[result_key])
                    )
            else:
                raise ValueError("Reduction type is sample or trajectory")

        # Add Episode Lengths to results
        metrics[
            "/".join([self._prefix, "episode_lengths"])
        ] = episode_lengths.float().to(accelerator.device)

        # Reset
        reset and self.reset()
        return metrics


class LoggingTrainingMetrics:
    def __init__(self, prefix="ppo"):
        self._metrics = defaultdict(list)
        self._prefix = prefix

    def add(self, key, value):
        self._metrics[key].append(value)

    def reset(self):
        self._metrics = defaultdict(list)

    def metrics_for_gather(self, accelerator):
        def tensor_mean(v):
            return torch.tensor(v).mean().float().to(accelerator.device)

        metrics = {
            "/".join([self._prefix, k]): tensor_mean(v)
            for k, v in self._metrics.items()
        }
        self.reset()
        return metrics
