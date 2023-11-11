from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

AUTOMODEL_CLASS = {
    "causal": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
}


class ModelType(Enum):
    CAUSAL = 0
    SEQ2SEQ = 1


@dataclass
class ActorOutput:
    """
    Dataclass for the output of the method policy.foward_policy()
    """

    # log probs corresponding to chosen actions
    log_probs: torch.tensor
    # entropy of action dist
    entropy: torch.tensor


@dataclass
class CriticOutput:
    """
    Dataclass for the output of the method policy.forward_value()
    """

    # values corresponding to given state
    values: torch.tensor


@dataclass
class ActorCriticOutput:
    """
    Dataclass for the output of the method policy.foward_policy()
    """

    # values of the given state
    values: torch.tensor
    # log prob of chosen actions
    log_prob: torch.tensor
    # entropy of action dist
    entropy: torch.tensor


@dataclass
class GenerationOutput:
    # log probs at each time step
    step_wise_logprobs: List[List[torch.tensor]]
    # actions at each time step
    step_wise_actions: List[torch.tensor]
    # generated tokens
    gen_tokens: List[List[int]]
    # generated texts
    gen_texts: List[str]
    # Sample Ids
    sample_ids: List[int]
