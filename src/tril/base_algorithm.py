from abc import ABC, abstractmethod
from typing import Optional

from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf

from tril.logging import Tracker


class BaseAlgorithm(ABC):
    """Abstract Class of algorithms.

    The basic structure of algorithms in TRIL. Once instantiated, Algorithm.learn()
    starts the learning loop. Implement specific setup, train_step, and eval_step
    to define learning behavior.
    """

    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        tracker: Optional[Tracker] = None,
    ):
        """Init for BaseAlgorithm.

        Args:
            cfg: full config passed in through Hydra
            accelerator: distributed computing accelerator object
            tracker: logger used for WandB logging, CLI logging, and model saving
        """

        # Accelerator
        self.accelerator = accelerator
        self.num_processes = accelerator.num_processes
        self.grad_accumulation_steps = accelerator.gradient_accumulation_steps

        # Logger
        self.tracker = tracker

        # Configs
        self.cfg = cfg
        self.alg_cfg = cfg.alg

        alg_attrs = OmegaConf.to_container(self.alg_cfg.args, resolve=True)
        for k, v in alg_attrs.items():
            setattr(self, k, v)

        self._setup()

    # Methods for model, dataset, optimizer, scheduler, and distributed computing setup
    @abstractmethod
    def _setup(self):
        """Setup method to initialize all objects and set up distributed computing.

        This method does the following in this order:
            - Instantiate task, agent, tokenizer: Instantiate components
            - call self._setup_dataloader: Instantiates dataloading pipelines
            - call self.prepare_accelerate: Sets up distributed computing strategy
        Note self._setup() is called in __init__
        """
        pass

    @abstractmethod
    def _setup_dataloaders(self):
        """Initializes dataloading sources for the algorithm.

        Example:
            self.buffer = OnlineBuffer()
            self.train_dataloader = torch.utils.data.DataLoader(self.buffer)
            self.val_dataloader = OfflineBuffer()
        """
        pass

    def _prepare_accelerate(self):
        """Calls respective prepare methods depending on distributed type.

        Currently suppoert FSDP and Deepspeed. Please refer to Accelerate
        documentation on how to setup Accelerate cfgs.
        """

        self.dist_type = self.accelerator.distributed_type
        if self.dist_type == DistributedType.DEEPSPEED:
            self._prepare_deepspeed()
        elif self.dist_type == DistributedType.FSDP:
            self._prepare_fsdp()
        else:
            raise ValueError(
                "Please set distributed computing type to either DEEPSPEED or FSDP"
            )

    @abstractmethod
    def _prepare_fsdp(self):
        """Prepare method for FSDP"""
        pass

    @abstractmethod
    def _prepare_deepspeed(self):
        """Prepare method for Deepspeed"""
        pass

    # Methods for Training and Evaluating
    @abstractmethod
    def train_step(self):
        """Training loop for algorithm."""
        pass

    @abstractmethod
    def eval_step(self):
        """Evaluation loop for algorithm"""
        pass

    @abstractmethod
    def learn(self):
        """Main algorithm loop combining training, evaluation, and sampling.

        Main methods to define the learning loop for the algorithm. For example,
        for RL pipelines, this method defines  a loop over:
            - Sampling
            - Train Step
            - Eval Step

        while for Supervised pipelines, we have
            - Train Step
            - Eval Step
        """
        pass
