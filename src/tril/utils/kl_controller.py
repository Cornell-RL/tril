from typing import Any, Dict, Optional

import torch
from torch.optim import Adam


class KLController:
    def __init__(
        self, kl_coeff: float, kl_lr: float = 0.1, target_kl: Optional[float] = None
    ) -> None:
        self._kl_coeff = kl_coeff
        self._target_kl = target_kl

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = (kl_div - self._target_kl) / self._target_kl
            # e_t = torch.clip(diff_to_target, -0.2, 0.2).item().
            # e_t = diff_to_target.item()
            e_t = diff_to_target
            self._kl_coeff = self._kl_coeff * (1 + 0.1 * e_t)

    @property
    def kl_coeff(self):
        return self._kl_coeff

    def get_state_dict(self) -> Dict[str, Any]:
        state = {"target_kl": self._target_kl, "current_kl_coeff": self._kl_coeff}
        return state

    def load_from_state_dict(self, state_dict: Dict[str, Any]):
        self._kl_coeff = state_dict["current_kl_coeff"]
        self._target_kl = state_dict["target_kl"]


"""
 TODO: Hardcoded scaled value
      64 - ppo batch
      100 - epochs
      7.5 - length of dataloader
"""


class ScaleKLController:
    def __init__(
        self, kl_coeff: float, kl_lr: float = 0.1, target_kl: Optional[float] = None
    ) -> None:
        self._kl_coeff = kl_coeff
        self._target_kl = target_kl

        self.batch_size = 64
        self.epochs = 100
        self.len_dataloader = 7.5

        self.horizon = self.epochs * self.len_dataloader

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = (kl_div - self._target_kl) / self._target_kl
            # e_t = torch.clip(diff_to_target, -0.2, 0.2).item()
            # e_t = diff_to_target.item()
            e_t = diff_to_target
            self._kl_coeff = self._kl_coeff * (1 + e_t * self.batch_size / self.horizon)

    @property
    def kl_coeff(self):
        return self._kl_coeff

    def get_state_dict(self) -> Dict[str, Any]:
        state = {"target_kl": self._target_kl, "current_kl_coeff": self._kl_coeff}
        return state

    def load_from_state_dict(self, state_dict: Dict[str, Any]):
        self._kl_coeff = state_dict["current_kl_coeff"]
        self._target_kl = state_dict["target_kl"]


class FixedKLController(KLController):
    def __init__(
        self, kl_coeff: float, kl_lr: float = 0.1, target_kl: Optional[float] = None
    ) -> None:
        self._kl_coeff = kl_coeff
        self._target_kl = target_kl

    def step(self, kl_div: torch.tensor):
        pass

    @property
    def kl_coeff(self):
        return self._kl_coeff

    def get_state_dict(self) -> Dict[str, Any]:
        state = {"target_kl": self._target_kl, "current_kl_coeff": self._kl_coeff}
        return state

    def load_from_state_dict(self, state_dict: Dict[str, Any]):
        self._kl_coeff = state_dict["current_kl_coeff"]
        self._target_kl = state_dict["target_kl"]


class KLPIDController(KLController):
    def __init__(
        self, kl_coeff: float, kl_lr: float = 0.1, target_kl: Optional[float] = None
    ) -> None:
        self._kl_coeff = torch.tensor([kl_coeff], requires_grad=True)
        self._target_kl = target_kl
        self._optim = Adam([self._kl_coeff], lr=kl_lr)

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = kl_div - self._target_kl  # / self._target_kl
            # e_t = torch.clip(diff_to_target, -0.2, 0.2).item()
            # e_t = diff_to_target.item()
            e_t = diff_to_target
            # self._kl_coeff = self._kl_coeff * (1 + 0.1 * e_t)
            self._kl_coeff.sum().backward()
            self._kl_coeff.grad = torch.tensor([-1 * e_t]).float()
            self._optim.step()

    @property
    def kl_coeff(self):
        return self._kl_coeff.detach().item()

    def get_state_dict(self) -> Dict[str, Any]:
        state = {"target_kl": self._target_kl, "current_kl_coeff": self._kl_coeff}
        return state

    def load_from_state_dict(self, state_dict: Dict[str, Any]):
        self._kl_coeff = state_dict["current_kl_coeff"]
        self._target_kl = state_dict["target_kl"]


class BallKLController(KLController):
    def __init__(
        self, kl_coeff: float, kl_lr: float = 0.1, target_kl: Optional[float] = None
    ) -> None:
        self._kl_coeff = torch.tensor([kl_coeff], requires_grad=True)
        self._target_kl = target_kl
        self._optim = Adam([self._kl_coeff], lr=kl_lr)

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = kl_div - self._target_kl
            # e_t = diff_to_target.item()
            e_t = diff_to_target
            # e_t = min(e_t, 0)

            self._kl_coeff.sum().backward()
            self._kl_coeff.grad = torch.tensor([-1 * e_t]).float()
            self._optim.step()

            # with torch.no_grad():
            #    self._kl_coeff.clamp_(max=0)
            # loss = -self._kl_coeff * (kl_div - self._target_kl)
            # self._optim.zero_grad(set_to_none=True)
            # loss.backward()
            # self._optim.step()

            # Constrain
            with torch.no_grad():
                self._kl_coeff.clamp_(min=0)

    @property
    def kl_coeff(self):
        return self._kl_coeff.detach().item()

    def get_state_dict(self) -> Dict[str, Any]:
        state = {"target_kl": self._target_kl, "current_kl_coeff": self._kl_coeff}
        return state

    def load_from_state_dict(self, state_dict: Dict[str, Any]):
        self._kl_coeff = state_dict["current_kl_coeff"]
        self._target_kl = state_dict["target_kl"]


class KLRegistry:
    _registry = {
        "klcontroller": KLController,
        "ballklcontroller": BallKLController,
        "klpidcontroller": KLPIDController,
        "fixedklcontroller": FixedKLController,
        "scaledklcontroller": ScaleKLController,
    }

    @classmethod
    def get(cls, kl_id: str):
        kl_cls = cls._registry[kl_id]
        return kl_cls

    @classmethod
    def add(cls, id: str, kl_cls):
        KLRegistry._registry[id] = kl_cls
