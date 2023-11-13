# TRIL Configs
The configs are organized into three main groups: `task`, `alg`, and `logging`. In `TRIL`, each `alg` config contains the specific hyperparameters for each task. For example:
```
# algorithm.yaml

task1:
  ...
task2:
  ...
```
When adding a new task, please add the respective task config in the algorithm config. Specifically, if you're adding a task with name `helpfulness`, and want to run `ppo`, add a `helpfulness:` field inside of `ppo.yaml`.  

## Logging Config: `logging`
The logging config is where you can define WandB/logger parameters.
```
log_to_wandb: false                             <- Flag for WandB Logging
entity_name: null                               <- WandB Entity
project_name: TRIL                              <- WandB Project
```

## Task Config: `task`
In the `task.yaml` there are four fields, `task`, `reward_fn`, `sampling` and `eval_metrics`. 

#### task
The `id` refers to the name in the [task registry](https://github.com/Cornell-RL/tril/blob/main/src/tril/tasks/__init__.py) while the `args` are task specific parameters defined in [tasks.py](https://github.com/Cornell-RL/tril/blob/main/src/tril/tasks/tasks.py).
```
task:                                           <- Task Config
  id: imdb                                      <- Task Name
  args:                                         <- Task Arguments
    seed: 42
```


#### reward_fn
The `id` refers to the name in the [reward_registry](https://github.com/Cornell-RL/tril/blob/main/src/tril/rewards/__init__.py) while the `args` are reward specific parameters. Note this field can be overwritten if defined in `alg`. For example, for GAIL, we have the reward be overwritten by [gail.yaml](https://github.com/Cornell-RL/tril/blob/main/cfgs/alg/gail.yaml).
```
reward_fn:                                      <- Reward Config
  id: learned_reward                            <- Reward Arguments
  args: 
    model_name: lvwerra/distilbert-imdb
    label_ix: 1
    include_prompt_for_eval: True
```

#### sampling
This field defines all the decoding and online data collection specific for each task. This involves, tokenizer details, generation lengths, per device sampling capacities, and train/eval decoding. Note `batch_size_per_process` denotes how many generations `model.generate()` will generate *per* process. For example, with four gpus and the following config, we would generate 448 trajectories per sampling iteration.
```
sampling:                                       <- Config for sampling/decoding of models
  batch_size_per_process: 112
  max_prompt_len: 64
  max_gen_len: 48
  prompt_padding_side: left
  prompt_truncation_side: left
  context_padding_side: right
  context_truncation_side: right
  train_generation_kwargs:                      <- Training Generation Arguments
    do_sample: True
    top_k: 50
    min_length: 48
    max_new_tokens: 48
  eval_generation_kwargs:                       <- Evaluation Generation Arguments
    do_sample: True
    top_k: 50
    min_length: 48
    max_new_tokens: 48
```

#### eval_metrics
Finally, define the evaluation metrics as a list for this task. Similar to reward_fn and task, the `id` is what the metric is registered under in the [metric registry](https://github.com/Cornell-RL/tril/blob/main/src/tril/metrics/__init__.py).
```
eval_metrics:                                   <- Metrics for Evaluation (List)
  - id: learned_reward
    args: 
      model_name: lvwerra/distilbert-imdb
      label_ix: 1
      batch_size: 100 
  - id: causal_perplexity
    args:
      tokenizer_id: gpt2
      stride: 512
      model_type: causal
```

## Algorithm Config: `alg`
This is where we define all algorithm specific parameters. Using `ppo.yaml` and `imdb` as an example here are the subfields.

#### Main section
We first define the `id` for the [algorithm registry](https://github.com/Cornell-RL/tril/blob/main/src/tril/algorithms/__init__.py) and `args` to instantiate our algorithm.
```
id: ppo                                         <- Algorithm ID
build_reward: True                              <- Build reward for algorithm

args:                                           <- Algorithm Training/Evaluation Arguments
  seed: 0
  verbose: 0                                    <- Logging Verbosity
  n_iters: 50                                   <- Total number of iterations to run `alg.learn`
  batch_size: 28                                <- Effective batch size (i.e. grad_accumulatin * devices * per_device)
  grad_accumulation: 1
  trajectories_per_update: 112                  <- Number of Trajectories/Generations per iteration
  n_epochs: 5                                   <- Number of Epochs to run within one iteration
  gamma: 0.99                                   <- Horizon Discount Term
  gae_lambda: 0.95                              <- Hyperparameter for Generalized Advantage Estimation
  vf_coef: 0.5                                  <- Critic/Value function loss coefficient
  target_coef: 0.1                              <- Target Regularization loss coefficient
  ent_coef: 0.0                                 <- Entropy Regularization loss coefficient
  target_regularization: true                   <- Flag for target regularization
  clip_range: 0.2                               <- Clip range for Policy Gradient
  clip_range_vf: 0.2                            <- Clip range for Critic Value
  max_grad_norm: 1.0                            <- Gradient Clipping
  target_kl: null                               <- Early Stopping Condition
  eval_batch_size: 100
  eval_every: 10
  save_every: 100
  eval_zero_shot: true
  save_checkpoints: false
  eval_splits: ['val']                          <- Splits to evaluation on: 'val' and/or 'test'
  max_prompt_len: ${sampling.max_prompt_len}
  max_gen_len: ${sampling.max_gen_len}
```

#### kl_div
This section, we define the parameters to construct the KL controller used to compute the reward penalty. The `kl_type` field is used to choose our controller.
```
kl_div:                                         <- KL Controller Arguments
  kl_type: 'fixedklcontroller'
  kl_lr: .01
  coeff: 0.001                                  <- KL coefficient for reward penalty
  target_kl: 0.1
```

#### optimizer and scheduler
Define the type of optimizer and scheduler as well as the optimizer specific parameters.
```
optimizer:                               <- Algorithm Optimizer Arguments
  id: adamw
  args:
    lr: 1e-5
    weight_decay: 1e-6
    eps: 1e-5

scheduler:                                      <- Optimizer Scheduler Arguments
  id: linear
  args:
    total_iters: 50
```

#### tokenizer
Here we define the tokenizer parameters that our LLM policy will use.
```
tokenizer:                                      <- Policy Tokenizer Arguments
  model_name: lvwerra/gpt2-imdb
  padding_side: left
  truncation_side: left
  pad_token_as_eos_token: True
```

#### policy
Here we construct the details of our LLM policy. For example, for PPO, we want to instantiate an `actor_critic` policy while for BC, we may just want an `actor` policy.
```
policy:                                         <- Policy Arguments
  id: actor_critic
  args:
    model_type: causal                          <- Either 'causal' or 'seq2seq'
    model_name: rajkumarrrk/gpt2-fine-tuned-on-imdb-positive-reviews
    max_prompt_len: ${sampling.max_prompt_len}
    max_gen_len: ${sampling.max_gen_len}
    create_reference: True                      <- Whether to create frozen reference model
    mlp_head: False                             <- MLP critic head
    quantize_model: False                       <- Flag for load in 4bit
    gen_kwargs: ${sampling.train_generation_kwargs}
    prompt_truncation_side: ${sampling.prompt_truncation_side}
```

## Full Config Example

Here is a full example config for a `PPO` run
```
# Logging Configs
experiment_name: tril_experiment                <- Experiment Name
log_to_wandb: false                             <- Flag for WandB Logging
entity_name: null                               <- WandB Entity
project_name: TRIL                              <- WandB Project

# Task Configs
task:                                           <- Task Config
  id: imdb                                      <- Task Name
  args:                                         <- Task Arguments
    seed: 42

reward_fn:                                      <- Reward Config
  id: learned_reward                            <- Reward Arguments
  args: 
    model_name: lvwerra/distilbert-imdb
    label_ix: 1
    include_prompt_for_eval: True

sampling:                                       <- Config for sampling/decoding of models
  batch_size_per_process: 112
  max_prompt_len: 64
  max_gen_len: 48
  prompt_padding_side: left
  prompt_truncation_side: left
  context_padding_side: right
  context_truncation_side: right
  train_generation_kwargs:                      <- Training Generation Arguments
    do_sample: True
    top_k: 50
    min_length: 48
    max_new_tokens: 48
  eval_generation_kwargs:                       <- Evaluation Generation Arguments
    do_sample: True
    top_k: 50
    min_length: 48
    max_new_tokens: 48

eval_metrics:                                   <- Metrics for Evaluation (List)
  - id: learned_reward
    args: 
      model_name: lvwerra/distilbert-imdb
      label_ix: 1
      batch_size: 100 
  - id: causal_perplexity
    args:
      tokenizer_id: gpt2
      stride: 512
      model_type: causal

# Algorithm Configs
id: ppo                                         <- Algorithm ID
build_reward: True                              <- Build reward for algorithm

args:                                           <- Algorithm Training/Evaluation Arguments
  seed: 0
  verbose: 0                                    <- Logging Verbosity
  n_iters: 50                                   <- Total number of iterations to run `alg.learn`
  batch_size: 28                                <- Effective batch size (i.e. grad_accumulatin * devices * per_device)
  grad_accumulation: 1
  trajectories_per_update: 112                  <- Number of Trajectories/Generations per iteration
  n_epochs: 5                                   <- Number of Epochs to run within one iteration
  gamma: 0.99                                   <- Horizon Discount Term
  gae_lambda: 0.95                              <- Hyperparameter for Generalized Advantage Estimation
  vf_coef: 0.5                                  <- Critic/Value function loss coefficient
  target_coef: 0.1                              <- Target Regularization loss coefficient
  ent_coef: 0.0                                 <- Entropy Regularization loss coefficient
  target_regularization: true                   <- Flag for target regularization
  clip_range: 0.2                               <- Clip range for Policy Gradient
  clip_range_vf: 0.2                            <- Clip range for Critic Value
  max_grad_norm: 1.0                            <- Gradient Clipping
  target_kl: null                               <- Early Stopping Condition
  eval_batch_size: 100 
  eval_every: 10
  save_every: 100
  eval_zero_shot: true
  save_checkpoints: false
  eval_splits: ['val']                          <- Splits to evaluation on: 'val' and/or 'test'
  max_prompt_len: ${sampling.max_prompt_len}
  max_gen_len: ${sampling.max_gen_len}

kl_div:                                         <- KL Controller Arguments
  kl_type: 'fixedklcontroller'
  kl_lr: .01
  coeff: 0.001                                  <- KL coefficient for reward penalty
  target_kl: 0.1 

optimizer:                               <- Algorithm Optimizer Arguments
  id: adamw
  args:
    lr: 1e-5
    weight_decay: 1e-6
    eps: 1e-5

scheduler:                                      <- Optimizer Scheduler Arguments
  id: linear
  args:
    total_iters: 50

tokenizer:                                      <- Policy Tokenizer Arguments
  model_name: lvwerra/gpt2-imdb
  padding_side: left 
  truncation_side: left 
  pad_token_as_eos_token: True 

policy:                                         <- Policy Arguments
  id: actor_critic
  args:
    model_type: causal                          <- Either 'causal' or 'seq2seq'
    model_name: rajkumarrrk/gpt2-fine-tuned-on-imdb-positive-reviews
    max_prompt_len: ${sampling.max_prompt_len}
    max_gen_len: ${sampling.max_gen_len}
    create_reference: True                      <- Whether to create frozen reference model
    mlp_head: False                             <- MLP critic head
    quantize_model: False                       <- Flag for load in 4bit
    gen_kwargs: ${sampling.train_generation_kwargs}
    prompt_truncation_side: ${sampling.prompt_truncation_side}
```
