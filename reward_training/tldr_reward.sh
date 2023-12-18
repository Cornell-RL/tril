#!/bin/bash
accelerate launch --config_file accelerate_cfgs/ds_rm.yaml --main_process_port 29520 --num_processes 4 reward_training/reward_trainer.py
