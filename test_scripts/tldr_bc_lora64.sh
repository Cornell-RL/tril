#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 4 main.py \
    task=tldr_supervised \
    alg=bc \
    alg.tldr.policy.args.model_name=EleutherAI/gpt-j-6b \
    alg.tldr.policy.args.lora.peft_config.r=64 \
    alg.tldr.policy.args.lora.peft_config.lora_alpha=128
