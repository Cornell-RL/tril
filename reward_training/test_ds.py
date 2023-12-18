import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os

os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ['DS_SKIP_CUDA_CHECK'] = '1'
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"] = "1"

import torch
import logging
from datetime import timedelta

import hydra
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf

from tril import tril_run
from tril.algorithms import AlgorithmRegistry
from tril.logging import Tracker
from tril.utils.builders import build_tokenizer
from tril.agent import Agent


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
@tril_run
def main(cfg: DictConfig):
    # init accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=720000))
    accelerator = Accelerator(
        dispatch_batches=False,
        gradient_accumulation_steps=cfg.alg.args.grad_accumulation,
        kwargs_handlers=[kwargs],
    )

    if accelerator.state.deepspeed_plugin is not None:
        if "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["fp16"][
                "auto_cast"
            ] = False

    tokenizer_cfg = cfg.alg.tokenizer
    tokenizer = build_tokenizer(tokenizer_cfg)
    agent = Agent(cfg, accelerator, tokenizer)
    optimizer = agent.setup_optimizer()
    agent, optimizer = accelerator.prepare(agent, optimizer)

    

    #for i in range(torch.cuda.device_count()):
    #    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(i)/1024/1024/1024))
    #    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(i)/1024/1024/1024))
    #    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(i)/1024/1024/1024))



if __name__ == "__main__":
    main()
