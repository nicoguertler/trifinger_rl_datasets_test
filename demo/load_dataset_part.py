"""Load part of a datset defined by a range of transitions."""


import gymnasium as gym

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    env = gym.make(
        "trifinger-cube-push-real-expert-image-mini-v0",
        disable_env_checker=True
    )

    # load only a subset of obervations, actions and rewards
    dataset = env.get_dataset(rng=(1000, 2000))

    n_observations = len(dataset["observations"])
    print("Number of observations: ", n_observations)

    assert dataset["actions"].shape[0] == n_observations
    assert dataset["rewards"].shape[0] == n_observations
