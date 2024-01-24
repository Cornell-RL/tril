#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29084 --num_processes 4 main.py \
    task=tldr \
    alg=pporef \
    experiment_name=ppo_ref_vf \
    alg.tldr_preference_rl.args.vf_coef=0.5
    #experiment_name=ppo_ref_costa \
    #task.args.reference_type=chosen \
    #alg.tldr_preference_rl.optimizer.args.lr=1e-5 \
    ##alg.tldr_preference_rl.kl_div.coeff=0.05

#accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 1 main.py task=tldr alg=pporef experiment
