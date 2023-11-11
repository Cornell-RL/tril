#!/bin/bash
accelerate launch --config_file accelerate_cfgs/fsdp_config.yaml --num_processes 1 main.py task=imdb alg=bc
