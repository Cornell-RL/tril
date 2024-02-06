CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 6 ppo_pp.py \
    --track \
    --beta=1.0 \
    --wandb_project_name=tldr_full \
    --exp_name=pythia_ppopp_lora_beta_1.0 \
    --output_dir=modelsv2/ppopp \
    --world_size=6 \
    --gradient_accumulation_steps=21 \
    --local_micro_batch_size=4 \
    --micro_batch_size=24 \
    --local_batch_size=84 \
    --batch_size=504


#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_ref.py
