<h1 align="center"> <p>TRIL</p></h1>
<h3 align="center">
    <p>Transformers Reinforcement and Imitation Learning Library</p>
</h3>

`TRIL` is a modular library for Reinforcment Learning (RL) and Imitation Learning (IL) algorithm development with transformers. We directly build on top of [`transformers`](https://github.com/huggingface/transformers), [`accelerate`](https://huggingface.co/docs/accelerate/index), and [`peft`](https://huggingface.co/docs/peft/index) libraries by 🤗 Hugging Face. That way TRIL is able to support open-sourced pretrained models, distributed computing, as well as parameter efficient training. Note we currently support most decoder and encoder-decoder architectures availble in `transformers`.

**Supported Algorithms:**

- Behavior Cloning (i.e. Supervised Fine Tuning)
- Proximal Policy Optimization (PPO) (https://arxiv.org/abs/1707.06347)
- Generative Adversarial Imitation Learning (GAIL) (https://arxiv.org/abs/1606.03476)
- PPO++ (https://arxiv.org/pdf/2306.11816)
- AggreVaTeD (https://arxiv.org/pdf/2306.11816)
- Locally Optimal Learning to Search (LOLS) (https://arxiv.org/pdf/2306.11816)
- Direct and Differentiable Locally Optimal Learning to Search (D2LOLS) (https://arxiv.org/pdf/2306.11816)

**Supported Tasks:**
- IMDB Positive Sentiment (https://arxiv.org/abs/2210.01241)
- CommonGen: Common Sense Generation (https://arxiv.org/abs/1911.03705)
- TL;DR Summarization (https://arxiv.org/pdf/2203.02155.pdf)

---

**Planned Algorithms:**
- Direct Preference Optimization (DPO) (https://arxiv.org/pdf/2305.18290.pdf)
- Statistical Rejection Sampling Optimization (RSO) (https://arxiv.org/pdf/2309.06657.pdf)
- Phasic Policy Gradient (PPG) (https://arxiv.org/abs/2009.04416)
- Pairwise Proximal Policy Optimization (P3O) (https://arxiv.org/pdf/2310.00212.pdf)
- Advantage-Induced Policy Alignment (APA) (https://arxiv.org/pdf/2306.02231.pdf)
- Advantage-Leftover Lunch RL (A-LoL) (https://arxiv.org/abs/2305.14718)

**Planned Tasks:**
- Helpfulness and Harmfullness (https://arxiv.org/pdf/2204.05862.pdf)


## Installation
To install `tril` do:
```
pip install tril
```
For the run scripts and the example scripts for usage please see the respository.

To setup a development environment we use `conda` for version control. To install TRIL, please follow these steps
```
conda create -n tril python=3.10
conda activate tril
pip install -e .
```

Optionally, for `caption_metrics` such as CiDER-D and SPICE, please install these additional dependencies.
```
# Spacy model install
python -m spacy download en_core_web_sm

# CoreNLP library install
cd src/tril/metrics/caption_metrics/spice && bash get_stanford_models.sh
```

## Example Scripts
In the `examples` directory, there are example scripts to run TRIL algorithms on `IMDB` positive sentiment generation using pytorch `Fully Sharded Data Parallel (FSDP)` and `TL;DR` summarization using `deepspeed`. The name of each script is of the format, `<task>_<alg>.yaml`. Run each experiment like the following:
```
./examples/<task>/<script>
```

Within each script the command is
```
accelerate --config <accelerate config> [accelerate args] main.py task=<task config> alg=<alg config> [hydra CLI config specification]
```

Please see the [`accelerate` launch tutorial](https://huggingface.co/docs/accelerate/basic_tutorials/launch) for how to launch jobs with `accelerate`. We provide examples of different `accelerate` configs in the `accelerate_cfgs` directoy. For more details on Hydra CLI and config usage please see this [tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).

## Usage Example
Here is a minimal example of running PPO with TRIL:
```python
import hydra
from accelerate import Accelerator
from tril import tril_run
from tril.logging import Tracker
from tril.algorithms import PPO

@hydra.main(version_base=None, config_path="cfgs", config_name="config") # Hydra Decorator for Config
@tril_run # TRIL decorator for hydra config processing
def run_ppo(cfg):
    # Initialize accelerator for distributed computing
    accelerator = Accelerator()

    # Grab experiment save directory from Hydra
    save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Instantiate TRIL logger for WandB and CLI logging/saving
    tracker = Tracker(
        save_path,
        OmegaConf.to_container(cfg, resolve=True),
        cfg.project_name,
        cfg.experiment_name,
        cfg.entity_name,
        cfg.log_to_wandb,
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )

    # Instantiate Algorithm
    ppo = PPO(cfg, accelerator, tracker)

    # Start learn to train LLM
    ppo.learn()

if __name__ == '__main__':
    run_ppo()
```

`TRIL` also provides an [`AlgorithmRegistry`](https://github.com/Cornell-RL/tril/blob/main/src/tril/algorithms/__init__.py) to instantiate algorithms. Please see our `main.py` to see how our scripts instantiate the algorithms. The list of available algorithms can be seen by the configs in `cfgs/task`.

## Current Task/Algorithm Support Matrix

| Algorithm  | IMDB | CommonGen | TL;DR |
|------------| ---- | ---- | ---- |
| PPO        | ✅ | ✅ | ✅ |
| PPO++      | ✅ | ✅ | ✅ |
| AggreVaTeD | ✅ | ✅ | ✅ |
| LOLS       | ✅ | ✅ | ✅ |
| D2LOLS     | ✅ | ✅ | ✅ |
| BC         | ✅ | ✅ | ✅ |
| GAIL       | ✅ |  |  |

## Code Structure
The directory structure of the configs, run script, and TRIL components looks like this.

```
├── cfgs                    <- Hydra configs
│   ├── alg                 <- Algorithm configs (e.g. PPO)
│   ├── task                <- Task configs (e.g. TL;DR summarization)
│   ├── logging             <- Logging configs (e.g. WandB)
│   │
│   └── config.yaml         <- Main config for training
│
├── accelerate_cfgs         <- Accelerate configs
│
├── main.py                 <- TRIL main function
│
├── tril                    <- TRIL src
│   ├── algorithms          <- Algorithm implementations
│   ├── buffers             <- Data Buffer (e.g. OnlineBuffer, PromptBuffer)
│   ├── metrics             <- Evaluation Metrics
│   ├── policies            <- Language Model Policies (e.g. Actor, ActorCritic)
│   ├── rewards             <- Reward Functions
│   ├── tasks               <- Supported Tasks
│   ├── utils               <- Helper functions for TRIL
│   │
│   ├── agent.py            <- Agent contains all torch.nn Modules (i.e. Policy and Reward)
│   ├── base_algorithm.py   <- Algorithm abstract class
│   ├── base_metric.py      <- Metric abstract class
│   ├── base_reward.py      <- Reward abstract class
│   ├── base_task.py        <- Task abstract class
│   └── logging.py          <- TRIL Logger
```

In each directory's `__init__.py`, there is a registry to register all supported `algorithms`, `metrics`, `rewards`, and `tasks`. When extending `TRIL`, please add the respective addition to one of these registries.

## Logging
TRIL support Weights and Biases logging. Please enter your `wandb` details such as `entity_name` and `project_name` into `cfgs/logging/wandb.yaml`. If you would not like to log to `wandb`, please set `log_to_wandb=False`.

By default, we save training and evaluation information in `outputs/<experiment_name>/<datetime>` You can define `experiment_name` in `cfgs/config.yaml` or through Hydra CLI, `main.py experiment_name=<name>`.


## Example WandB Reports
Here is an example WandB Report of how the logging would look like when running multiple different algorithms

* [CommonGen Report](https://api.wandb.ai/links/coactivelearning/hfocjp17).
* [TL;DR PPO Report](https://api.wandb.ai/links/coactivelearning/ga4r1uqd).

## Citing TRIL
If you use TRIL in your publication, please cite it by using the following BibTeX entry.
```bibtex
@misc{TRIL,
      title={TRIL: Transformers Reinforcement and Imitation Learning Library},
      author={Jonathan D Chang and Kiante Brantley and Rajkumar Ramamurthy and Dipendra Misra and Wen Sun},
      howpublished={\url{https://github.com/Cornell-RL/tril}},
      year={2023}
}
```

Here is the citation of the accompanying paper for many of the supported algorithms.
```bibtex
@misc{chang2023learning,
      title={Learning to Generate Better Than Your LLM}, 
      author={Jonathan D. Chang and Kiante Brantley and Rajkumar Ramamurthy and Dipendra Misra and Wen Sun},
      year={2023},
      eprint={2306.11816},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
We would like to acknowledge [RL4LMs](https://github.com/allenai/RL4LMs), [TRL](https://github.com/huggingface/trl), and [TRLx](https://github.com/CarperAI/trlx) for being inspirations for this library.
