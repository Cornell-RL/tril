#!/bin/bash
accelerate launch --config_file accelerate_cfgs/fsdp_config.yaml --main_process_port 29636 --num_processes 1 main.py task=imdb alg=gail
