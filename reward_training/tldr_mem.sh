#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 4 test_ds.py task=tldr alg=ppo
