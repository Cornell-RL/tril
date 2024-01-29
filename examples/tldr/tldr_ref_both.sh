#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29095 --num_processes 4 main.py \
    task=tldr \
    alg=pporef \
    task.args.reference_type=both \
    experiment_name=ppo_ref_with_both

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 main.py task=tldr alg=pporef