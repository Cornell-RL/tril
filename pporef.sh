accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py \
    --track \
    --beta=1.0 \
    --wandb_project_name=tldr_full \
    --exp_name=pythia_pporef_lora_beta_1.0 \
    --output_dir=modelsv2/pporef \
    --world_size=8 \
    --gradient_accumulation_steps=16 \
    --local_micro_batch_size=4 \
    --micro_batch_size=32 \
    --local_batch_size=64 \
    --batch_size=512


#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_ref.py
