accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 8 sft.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 sft.py
