#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29090 --num_processes 4 main.py \
    task=tldr \
    alg=pporef \
    task.args.reference_type=rejected \
    experiment_name=ppo_ref_with_rejected

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 main.py task=tldr alg=pporef
