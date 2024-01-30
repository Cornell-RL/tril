accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 4 ppo_ref.py \
    --track \
    --beta=0.2 \
    --exp_name=pythia_pporef_lora_beta_0.2 \
    --output_dir=models/pporef_model_new_0.2 \
    --world_size=4 \
    --gradient_accumulation_steps=32 \
    --local_micro_batch_size=4 \
    --micro_batch_size=16 \
    --local_batch_size=128 \
    --batch_size=512


#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_ref.py
