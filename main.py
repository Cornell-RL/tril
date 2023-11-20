import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os

os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ['DS_SKIP_CUDA_CHECK'] = '1'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"] = "1"

import logging
from datetime import timedelta

import hydra
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf

from tril import tril_run
from tril.algorithms import AlgorithmRegistry
from tril.logging import Tracker
from tril.utils.multi_engine_accelerator import DeepspeedMultiEngineAccelerator


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
@tril_run
def main(cfg: DictConfig):
    # init accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=720000))
    #accelerator = Accelerator(
    accelerator = DeepspeedMultiEngineAccelerator(
        dispatch_batches=False,
        gradient_accumulation_steps=cfg.alg.args.grad_accumulation,
        kwargs_handlers=[kwargs],
    )

    if accelerator.state.deepspeed_plugin is not None:
        if "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["fp16"][
                "auto_cast"
            ] = False

    save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tracker = Tracker(
        save_path,
        OmegaConf.to_container(cfg, resolve=True),
        cfg.project_name,
        cfg.experiment_name,
        cfg.entity_name,
        cfg.log_to_wandb,
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )

    # Initialize
    try:
        alg_cls = AlgorithmRegistry.get(cfg.alg.id)
    except:
        raise NotImplementedError(
            f"Algorithm {cfg.alg.id} is not supported yet. If implemented, please regist in 'tril.algorithms'."
        )

    alg = alg_cls(cfg, accelerator, tracker)

    # Start Program
    alg.learn()


if __name__ == "__main__":
    main()
