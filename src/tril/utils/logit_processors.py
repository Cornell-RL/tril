import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.functional import one_hot
from transformers.generation.logits_process import LogitsProcessor


class RawLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.raw_logits = []

    def get_logits(self):
        return self.raw_logits

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.raw_logits.append(scores.clone())
        return scores


class TeacherForcingProcessor(LogitsProcessor):
    def __init__(self, actions):
        self.counter = 0
        self.actions = actions  # (Batch_size, max_seq_len, 1)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.size(-1)
        new_scores = one_hot(
            self.actions[:, self.counter], num_classes=vocab_size
        ).float()
        assert scores.shape == new_scores.shape  # (batch_size, vocab_size)
        new_scores = new_scores.to(scores.device)
        self.counter += 1
        return new_scores


class TeacherForcingLogProbProcessor(LogitsProcessor):
    def __init__(self, actions):
        self.counter = 0
        self.actions = actions  # (Batch_size, max_seq_len, 1)
        self.log_probs = []

    def get_log_probs(self):
        return torch.concatenate(
            [x.unsqueeze(1) for x in self.log_probs], dim=1
        )  # (batch, seq_len, 1)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.size(-1)
        new_scores = one_hot(
            self.actions[:, self.counter], num_classes=vocab_size
        ).float()
        assert scores.shape == new_scores.shape  # (batch_size, vocab_size)
        new_scores = new_scores.to(scores.device)
        dist = Categorical(logits=scores)
        log_probs = dist.log_prob(self.actions[:, self.counter])  # (batch_size, 1)
        self.log_probs.append(log_probs.cpu())
        self.counter += 1
        return new_scores


class RollinProcessor(LogitsProcessor):
    def __init__(self, actions, beta, rng, seq_lens):
        self.counter = 0
        self.actions = actions  # (Batch_size, max_seq_len, 1)
        self.seq_lens = seq_lens
        self.create_mask(beta, rng)

    def create_mask(self, beta, rng):
        batch_size, seq_len = self.actions.shape[:2]
        # Mixin
        init_mask = rng.choice([True, False], size=(batch_size, 1), p=[beta, 1 - beta])
        init_mask = np.tile(init_mask, (1, seq_len))
        # Rollin Selection
        length_masks = np.tril(np.ones((seq_len, seq_len)))
        masks = []
        # Do this sequentially so we do not sample rollin_length > sequence length
        for length in self.seq_lens:
            if length < 2:
                masks.append(np.zeros((seq_len)).astype(bool))
            else:
                masks.append(rng.choice(length_masks[: length - 1, :]).astype(bool))

        self.rollin_mask = np.stack(masks)
        self.rollin_mask[~init_mask] = False
        # self.rollin_mask = np.zeros((3072, 20)).astype(bool)

    def get_rollin_mask(self):
        return torch.tensor(self.rollin_mask)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.size(-1)
        new_scores = one_hot(
            self.actions[:, self.counter], num_classes=vocab_size
        ).float()
        new_scores[new_scores == 0] = -float("inf")
        mask = self.rollin_mask[:, self.counter]
        assert scores.shape == new_scores.shape  # (batch_size, vocab_size)

        new_scores = new_scores.to(scores.device)
        # Only do Teacher Forcing on the rollins
        scores[mask] = new_scores[mask]
        # import pdb; pdb.set_trace()
        # new_scores[~mask] = scores[~mask]
        self.counter += 1
        # return new_scores
        return scores


class ValueProcessor(LogitsProcessor):
    def __init__(self, value_head, accelerator):
        self.values = []
        self.value_head = value_head
        # self.device = torch.device('cuda')
        self.accelerator = accelerator
        # self.hidden_states = []

    def get_values(self):
        return torch.concatenate(
            [x.unsqueeze(1) for x in self.values], dim=1
        )  # (batch, seq_len, 1)

    # TODO: improve this with model creation
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        model_inputs,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # self.hidden_states.append(hidden_states[-1].squeeze().cpu())
        with torch.no_grad():
            # value = self.value_head(hidden_states[-1].squeeze().to(self.device))
            value = self.value_head(
                hidden_states[-1].squeeze().to(self.accelerator.device)
            )
        self.values.append(value.cpu())
        return scores


class ValueScoreProcessor(LogitsProcessor):
    def __init__(self, value_head, accelerator):
        self.values = []
        self.value_head = value_head
        # self.device = torch.device('cuda')
        self.accelerator = accelerator
        # self.hidden_states = []

    def get_values(self):
        return torch.concatenate(
            [x.unsqueeze(1) for x in self.values], dim=1
        )  # (batch, seq_len, 1)

    # TODO: improve this with model creation
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        model_inputs,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # self.hidden_states.append(hidden_states[-1].squeeze().cpu())
        with torch.no_grad():
            # value = self.value_head(hidden_states[-1].squeeze().to(self.device))
            value = self.value_head(
                hidden_states[-1].squeeze().to(self.accelerator.device)
            )
        # get a value one hot
        self.values.append(value.cpu())
        return scores
