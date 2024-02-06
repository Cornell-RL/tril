accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 8 ppo_default.py \
    --track \
    --wandb_project_name=tldr_full \
    --exp_name=pythia_ppo_lora \
    --output_dir=modelsv2/ppo \
    --world_size=8 \
    --gradient_accumulation_steps=16 \
    --local_micro_batch_size=4 \
    --micro_batch_size=32 \
    --local_batch_size=64 \
    --batch_size=512
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_default.py
