#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 4 main.py task=tldr_supervised alg=bc experiment_name=sft
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 1 main.py task=tldr_supervised alg=bc experiment_name=sft log_to_wandb=false
