__version__ = "0.9.0"

from gymnasium.envs.registration import register

from .dataset_env import TriFingerDatasetEnv
from .evaluation import Evaluation
from .policy_base import PolicyBase, PolicyConfig


dataset_params = [
    # simulation
    # =========
    # real robot
    # ==========
    # push expert with images (mini version for testing)
    {
        "name": "trifinger-cube-push-real-expert-image-mini-v0",
        "dataset_url": "https://robots.real-robot-challenge.com/public/split_test/trifinger-cube-push-real-expert-image-mini-v0.yaml",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 15000 / 20,
        "real_robot": True,
        "trifinger_kwargs": {
            "episode_length": 750,
            "difficulty": 1,
            "keypoint_obs": True,
            "obs_action_delay": 10,
        },
    },
]


def get_env(**kwargs):
    return TriFingerDatasetEnv(**kwargs)


for params in dataset_params:
    register(
        id=params["name"], entry_point="trifinger_rl_datasets:get_env", kwargs=params
    )


__all__ = ("TriFingerDatasetEnv", "Evaluation", "PolicyBase", "PolicyConfig", "get_env")
