from gym.envs.registration import register

from .dataset_env import TriFingerDatasetEnv


dataset_params = [
    # simulation
    # =========
    # pushing-expert
    {
        "name": "trifinger-cube-push-sim-expert-v0",
        "dataset_url": "https://owncloud.tuebingen.mpg.de/index.php/s/XKcRys7JLPTyaqx/download",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 15000 / 20,
        "trifinger_kwargs": {
            "episode_length": 750,
            "difficulty": 1,
            "keypoint_obs": True,
            "obs_action_delay": 10,
        },
    }
    # real robot
    # ==========
]


def get_env(**kwargs):
    return TriFingerDatasetEnv(**kwargs)


for params in dataset_params:
    register(id=params["name"], entry_point="rrc_2022_datasets:get_env", kwargs=params)
