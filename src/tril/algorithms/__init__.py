from tril.algorithms.aggrevated import AGGREVATED
from tril.algorithms.bc import BC
from tril.algorithms.d2lols import D2LOLS
from tril.algorithms.gail import GAIL
from tril.algorithms.lols import LOLS
from tril.algorithms.ppo import PPO
from tril.algorithms.ppo_pp import PPO_PP


class AlgorithmRegistry:
    _registry = {
        "ppo": PPO,
        "ppo_pp": PPO_PP,
        "aggrevated": AGGREVATED,
        "lols": LOLS,
        "d2lols": D2LOLS,
        "bc": BC,
        "gail": GAIL,
    }

    @classmethod
    def get(cls, alg_id: str):
        try:
            alg_cls = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return alg_cls

    @classmethod
    def add(cls, id: str, alg_cls):
        AlgorithmRegistry._registry[id] = alg_cls
