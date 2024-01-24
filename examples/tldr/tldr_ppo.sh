#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 4 main.py \
    task=tldr \
    alg=ppo \
    experiment_name=ppo_truncate \
    log_to_wandb=false
    #alg.tldr_preference_rl.args.vf_coef=0.5
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 1 main.py task=tldr alg=ppo
