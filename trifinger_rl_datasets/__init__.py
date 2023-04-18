__version__ = "0.9.0"

from gymnasium.envs.registration import register

from .dataset_env import TriFingerDatasetEnv
from .evaluation import Evaluation
from .policy_base import PolicyBase, PolicyConfig


# simulation
# ==========
push_sim = [
    {
        "name": "trifinger-cube-push-sim-expert-v0",
        "dataset_url": "https://robots.real-robot-challenge.com/public/trifinger_rl_datasets/trifinger-cube-push-sim-expert-v0.zarr/dataset.yaml",
    },
    {
        "name": "trifinger-cube-push-sim-expert-image-v0",
        "dataset_url": "https://robots.real-robot-challenge.com/public/trifinger_rl_datasets/trifinger-cube-push-sim-expert-image-v0.zarr/dataset.yaml",
    }
]
lift_sim = []
# real robot
# ==========
push_real = [
    {
        "name": "trifinger-cube-push-real-expert-image-v0",
        "dataset_url": "https://robots.real-robot-challenge.com/public/trifinger_rl_datasets/trifinger-cube-push-real-expert-image-v0.zarr/dataset.yaml",
    }
]
lift_real = [
    {
        "name": "trifinger-cube-lift-real-expert-image-v0",
        "dataset_url": "https://robots.real-robot-challenge.com/public/trifinger_rl_datasets/trifinger-cube-lift-real-expert-image-v0.zarr/dataset.yaml",
    }
]

# add the missing parameters for all environments
dataset_params = []
for core_params in push_sim:
    dataset_params.append(
        {
            **core_params,
            "ref_min_score": 0.0,
            "ref_max_score": 1.0 * 15000 / 20,
            "real_robot": False,
            "trifinger_kwargs": {
                "episode_length": 750,
                "difficulty": 1,
                "keypoint_obs": True,
                "obs_action_delay": 10,
            },
        }
    )

for core_params in lift_sim:
    dataset_params.append(
        {
            **core_params,
            "ref_min_score": 0.0,
            "ref_max_score": 1.0 * 30000 / 20,
            "real_robot": False,
            "image_obs": True,
            "trifinger_kwargs": {
                "episode_length": 1500,
                "difficulty": 4,
                "keypoint_obs": True,
                "obs_action_delay": 2,
            },
        }
    )

for core_params in push_real:
    dataset_params.append(
        {
            **core_params,
            "ref_min_score": 0.0,
            "ref_max_score": 1.0 * 15000 / 20,
            "real_robot": True,
            "trifinger_kwargs": {
                "episode_length": 750,
                "difficulty": 1,
                "keypoint_obs": True,
                "obs_action_delay": 10,
            },
        }
    )

for core_params in lift_real:
    dataset_params.append(
        {
            **core_params,
            "ref_min_score": 0.0,
            "ref_max_score": 1.0 * 30000 / 20,
            "real_robot": True,
            "image_obs": True,
            "trifinger_kwargs": {
                "episode_length": 1500,
                "difficulty": 4,
                "keypoint_obs": True,
                "obs_action_delay": 2,
            },
        }
    )


def get_env(**kwargs):
    return TriFingerDatasetEnv(**kwargs)


for params in dataset_params:
    register(
        id=params["name"], entry_point="trifinger_rl_datasets:get_env", kwargs=params
    )


__all__ = ("TriFingerDatasetEnv", "Evaluation", "PolicyBase", "PolicyConfig", "get_env")
