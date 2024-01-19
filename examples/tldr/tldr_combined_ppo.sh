#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29086 --num_processes 4 main.py \
    task=tldr_combined \
    alg=ppo \
    experiment_name=ppo_tldr_combined
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 1 main.py task=tldr alg=ppo
