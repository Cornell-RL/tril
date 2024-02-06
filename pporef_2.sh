accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29089 --num_processes 2 ppo_ref.py \
    --track \
    --beta=0.75 \
    --exp_name=pythia_pporef_lora_beta_0.75 \
    --output_dir=models/pporef_model_new_0.75 \
    --world_size=2 \
    --gradient_accumulation_steps=64 \
    --local_micro_batch_size=4 \
    --micro_batch_size=8 \
    --local_batch_size=256 \
    --batch_size=512


#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_ref.py
