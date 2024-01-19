#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29096 --num_processes 4 main.py \
    task=tldr_combined \
    alg=pporef \
    task.args.reference_type=both \
    experiment_name=ppo_tldr_combined_ref_with_both

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 main.py task=tldr alg=pporef
