#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 1 main.py task=tldr alg=ppo
