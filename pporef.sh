#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 4 ppo_ref.py
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_ref.py
