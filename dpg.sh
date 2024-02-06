accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port 29085 --num_processes 4 dpg.py \
    --track \
    --exp_name=pythia_dpg_kl_0_eta_1 \
    --ppo.noptepochs=4 \
    --eta 1.0
