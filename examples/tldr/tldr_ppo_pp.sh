#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 4 main.py task=tldr alg=ppo_pp
