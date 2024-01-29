#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 4 main.py \
    task=tldr_openai \
    alg=ppo \
    experiment_name=ppo_openai_truncation_fix \
    project_name=tldr_ppo \
    log_to_wandb=true
    #alg.tldr_openai.optimizer.args.lr=1e-5 \
