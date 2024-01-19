#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29095 --num_processes 8 main.py \
    task=tldr \
    alg=pporef \
    alg.tldr.args.grad_accumulation=8
