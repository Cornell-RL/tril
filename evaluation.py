import os
import json
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights, set_peft_model_state_dict
from rm2 import ScalarModel, ScalarModelConfig

INVALID_LOGPROB = 1.0

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    use_adaptive_kl: bool = False
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    dataset_std: float = 1.0
    kl_coef: float = 0.05


@dataclass
class PpoHParams:
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
    noptepochs: int = 4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    #query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_pythia-160m_53"
    query_dataset: str = "cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162" # pythia 2.9

    query_format_str: Optional[str] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    query_truncate_field: Optional[str] = "post"
    query_truncate_text: Optional[str] = "\n"
    query_padding: Optional[str] = None  # defaults to repeated spaces
    query_pad_side: Optional[str] = "left"

    # Response params
    response_length: int = 53

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    truncate_after: int = 16
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.7


# a patch
@dataclass
class TaskQueryHParams:
    length: int = None
    dataset: str = None
    format_str: Optional[str] = None  # if underlying dataset yields dicts, can format arbitrarily
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[str] = None  # defaults to repeated spaces
    pad_side: Optional[str] = None


@dataclass
class Args:
    alg: str = "ppo_ref"
    adapter_path: str = "./models/pporef_model_new_1.0_660"
    # common args
    exp_name: str = "evaluation"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    #track: bool = True
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize_costa_lora"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "rollin_ref"
    """the entity (team) of wandb's project"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = True
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    world_size: Optional[int] = 4
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 32
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 4
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    #total_episodes: Optional[int] = 300000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = 16
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = 128
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = 512
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = 128
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = 512
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 32
    """per rank no grad forward pass in the rollout phase"""

    # other args
    #base_model: str = "EleutherAI/pythia-160m"
    base_model: str = "jdchang/tldr_sft_pythia_2.8"
    """the name of the pretrained model to use"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    #reward_model_path: str = "./models/reward_model_2.9"
    reward_model_path: str = "./models/reward_model_2.8_full"
    """the name of the pretrained model to use"""
    sft_model_path: str = "jdchang/tldr_sft_pythia_2.8"
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    output_dir: str = "models/ppo_model"
    """Where to save the model"""
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)

    label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"

# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses, is_cnn=False):
    if not is_cnn:
        l = args.task.response_length
    else:
        l = 155
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [l]
    idxs = torch.arange(l, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer, ref=False):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    if ref:
        with model.disable_adapter():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    else:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )


@dataclass
class EvalStorage:
    query_token: List[str] = field(default_factory=list)
    postprocessed_response_token: List[str] = field(default_factory=list)
    reference_response_token: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    reference_score: List[float] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list)

    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    reference_response: List[str] = field(default_factory=list)


def evaluate_policy(args: Args, reward_model, policy, tokenizer, dataloader, generation_config, is_cnn=False):
    eval_storage = EvalStorage()
    ref_key = "chosen_token" if is_cnn else "reference_response_token"
    with torch.no_grad():
        for data in tqdm(dataloader):
            queries = data["query_token"]
            reference_response_token = data[ref_key]
            context_length = queries.shape[1]
            query_reference_responses = torch.cat((data["query_token"], data[ref_key]), dim=1)
            #_, reference_score, _ = get_reward(reward_model, query_reference_responses, tokenizer, queries.shape[1])

            query_responses = generate(
                policy,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses, is_cnn=is_cnn)
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            gen_mask = postprocessed_responses != tokenizer.pad_token_id

            reference_responses = query_reference_responses[:, context_length:]
            postprocessed_reference_responses = truncate_response(args, tokenizer, reference_responses, is_cnn=is_cnn)
            postprocessed_reference_query_responses = torch.cat((queries, postprocessed_responses), 1)
            ref_mask = postprocessed_reference_responses != tokenizer.pad_token_id

            # Reward
            #_, score, _ = get_reward(reward_model, postprocessed_query_responses, tokenizer, queries.shape[1])

            # KL
            output = forward(policy, query_responses, tokenizer)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= args.task.temperature + 1e-7
            all_logprob = F.log_softmax(logits, dim=-1)
            logprob = torch.gather(all_logprob, 2, responses.unsqueeze(-1)).squeeze(-1)
            logprob = (logprob * gen_mask).sum(-1)

            #prob_dist = torch.nn.functional.softmax(logits, dim=-1)
            #entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
            #del output, logits, all_logprob, prob_dist
            del output, logits, all_logprob
            torch.cuda.empty_cache()

            ref_output = forward(policy, query_responses, tokenizer, ref=True)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= args.task.temperature + 1e-7
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            ref_logprob = torch.gather(ref_all_logprob, 2, responses.unsqueeze(-1)).squeeze(-1)
            ref_logprob = (ref_logprob * ref_mask).sum(-1)
            del ref_output, ref_logits, ref_all_logprob
            torch.cuda.empty_cache()
            score = 0.05 * (logprob - ref_logprob)
            #kl = (logprob - ref_logprob).sum(dim=1) # Average across sequences

            # Perplexity
            #ref_output = forward(policy, query_reference_responses, tokenizer, ref=True)
            #ref_logits = ref_output.logits[:, context_length - 1 : -1]
            #ref_logits /= args.task.temperature + 1e-7
            #ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            #perplexity_ref_logprob = torch.gather(ref_all_logprob, 2, reference_response_token.unsqueeze(-1)).squeeze(-1)
            #perplexity = torch.exp(-perplexity_ref_logprob.mean(dim=1))
            #del ref_output, ref_logits, ref_all_logprob, perplexity_ref_logprob
            #torch.cuda.empty_cache()


            # Store
            eval_storage.query_token.extend(queries)
            eval_storage.reference_response_token.extend(reference_response_token)
            #eval_storage.reference_score.append(reference_score)
            eval_storage.postprocessed_response_token.extend(postprocessed_responses)
            eval_storage.score.append(score)
            #eval_storage.kl.append(kl)
            #eval_storage.entropy.append(entropy)
            #eval_storage.perplexity.append(perplexity)

    eval_storage.query = tokenizer.batch_decode(eval_storage.query_token, skip_special_tokens=True)
    eval_storage.reference_response = tokenizer.batch_decode(eval_storage.reference_response_token, skip_special_tokens=True)
    eval_storage.postprocessed_response = tokenizer.batch_decode(
        eval_storage.postprocessed_response_token, skip_special_tokens=True
    )
    eval_score = torch.cat(eval_storage.score).float().cpu().numpy().tolist()
    #eval_reference_score = torch.cat(eval_storage.reference_score).float().cpu().numpy().tolist()
    #eval_kl = torch.cat(eval_storage.kl).float().cpu().numpy().tolist()
    #eval_perplexity = torch.cat(eval_storage.perplexity).float().cpu().numpy().tolist()
    #eval_entropy = torch.cat(eval_storage.entropy).float().cpu().numpy().tolist()


    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "reference_responses": gather_object(eval_storage.reference_response),
            "scores": gather_object(eval_score),
            #"reference_scores": gather_object(eval_reference_score),
            #"kl": gather_object(eval_kl),
            #"perplexity": gather_object(eval_perplexity),
            #"entropy": gather_object(eval_entropy),
        }
    )
    return eval_df


if __name__ == "__main__":
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.task.truncate_token == "eos":
        args.task.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    #if accelerator.is_main_process:
        #if args.track:
        #    import wandb

        #    wandb.init(
        #        project=args.wandb_project_name,
        #        entity=args.wandb_entity,
        #        sync_tensorboard=True,
        #        config=asdict(args),
        #        name=run_name,
        #        save_code=True,
        #    )
        #    file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
        #    wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        #writer = SummaryWriter(f"runs/{run_name}")
        #writer.add_text(
        #    "hyperparameters",
        #    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        #)
        #pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained(args.base_model)
    configure_dropout(model_config, args.dropout_layer_keys, 0.0)  # disable dropout
    scalar_model_config = ScalarModelConfig(
        base_model=args.base_model,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if not args.reward_model_path:
        reward_model: PreTrainedModel = ScalarModel(scalar_model_config)
    else:
        reward_model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    if accelerator.is_main_process:
        pprint(model_config)
        pprint(reward_model.config)
    # each class should have a separate pretrained model that do not share weights
    #ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)
    peft_config = LoraConfig(
        r=1024,
        lora_alpha=2048,
        #lora_dropout=0.05,
        lora_dropout=0.0,
        bias="none",
    )
    policy = get_peft_model(policy, peft_config=peft_config)

    # Load Pretrained Adapter
    adapter_state_dict = load_peft_weights(args.adapter_path, device=torch.device('cpu'))
    set_peft_model_state_dict(policy, adapter_state_dict)

    accelerator.print(policy)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    optimizer = optim.AdamW(policy.parameters())

    # Create Test Datasets
    eval_datasets = []
    eval_dataloaders = {}

    # TL;DR
    test_dataset = load_dataset(args.task.query_dataset, split="test")
    test_dataset = test_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    eval_datasets.append(test_dataset)
    eval_dataloaders["test"] = DataLoader(test_dataset, batch_size=args.local_eval_batch_size)

    # CNN/DM
    cnn_dataset = load_dataset(args.label_dataset, split="validation_cnndm")
    cnn_dataset = cnn_dataset.with_format("torch", columns=["query_token", "chosen_token"])
    eval_datasets.append(cnn_dataset)
    eval_dataloaders["test_cnndm"] = DataLoader(cnn_dataset, batch_size=args.local_eval_batch_size)

    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }


    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)  # reset the local seed again
    reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
    reward_model.eval()

    # Get inputs ready
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    cnn_generation_config = GenerationConfig(
        max_new_tokens=155,
        min_new_tokens=155,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    import evaluate
    rouge = evaluate.load('rouge')

    root_dir = f"final_results/{args.alg}"
    os.makedirs(root_dir, exist_ok=True)

    #for split in ["test", "test_cnndm"]:
    for split in ["test"]:
        with torch.no_grad():
            is_cnn = split != "test"
            if is_cnn:
                eval_df = evaluate_policy(
                    args,
                    reward_model,
                    accelerator.unwrap_model(policy),
                    tokenizer,
                    eval_dataloaders[split],
                    cnn_generation_config,
                    is_cnn=is_cnn
                )
            else:
                eval_df = evaluate_policy(
                    args,
                    reward_model,
                    accelerator.unwrap_model(policy),
                    tokenizer,
                    eval_dataloaders[split],
                    generation_config,
                    is_cnn=is_cnn
                )

            if accelerator.is_main_process:
                #generations = eval_df['postprocessed_response'].tolist()
                #references = eval_df['reference_responses'].tolist()
                #rouge_scores = rouge.compute(predictions=generations, references=references)
                eval_df.to_csv(root_dir + f"/{split}_table.csv")
                #with open(root_dir + f"/{split}_rouge.json", 'w+') as f:
                #    json.dump(rouge_scores, f)
            accelerator.wait_for_everyone()

