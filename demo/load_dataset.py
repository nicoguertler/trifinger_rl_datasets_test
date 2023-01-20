import gymnasium as gym

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    env = gym.make(
        "trifinger-cube-push-sim-expert-v0",
        disable_env_checker=True,
        visualization=True,  # enable visualization
    )
    dataset = env.get_dataset()

    n_transitions = len(dataset["observations"])
    print("Number of transitions: ", n_transitions)

    assert dataset["actions"].shape[0] == n_transitions
    assert dataset["rewards"].shape[0] == n_transitions

    print("First observation: ", dataset["observations"][0])

    obs = env.reset()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
