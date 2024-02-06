accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29090 --num_processes 8 evaluation.py \
    --alg=dpo_scores \
    --adapter_path=models/dpo_policy_model_2.8_lora \
    --exp_name=eval_sft \
    --gradient_accumulation_steps=1 \
    --local_micro_batch_size=2 

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29091 --num_processes 6 evaluation.py \
#    --alg=pporef_75 \
#    --adapter_path=models/pporef_model_new_0.75 \
#    --exp_name=eval_sft \
#    --gradient_accumulation_steps=1 \
#    --local_micro_batch_size=2 

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29092 --num_processes 6 evaluation.py \
#    --alg=pporef_100_880 \
#    --adapter_path=models/pporef_model_new \
#    --exp_name=eval_sft \
#    --gradient_accumulation_steps=1 \
#    --local_micro_batch_size=2 

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29087 --num_processes 8 ppo_ref.py
#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 ppo_ref.py
