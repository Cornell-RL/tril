#!/bin/bash

accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29080 --num_processes 8 rm2.py \
    --exp_name=rm20 \
    --num_train=18566 \
    --output_dir=models/rm20

accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29082 --num_processes 8 rm2.py \
    --exp_name=rm60 \
    --num_train=55700 \
    --output_dir=models/rm60
