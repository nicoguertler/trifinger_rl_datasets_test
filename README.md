# TriFinger RL Datasets

This repository provides offline reinforcement learning datasets collected on the TriFinger platform (simulated or real). The paper "Benchmarking Offline Reinforcement Learning on Real-Robot Hardware" (TODO: Link) discusses the datasets and benchmarks offline reinforcement learning methods on them. The code for loading the datasets follows the interface suggested by [D4RL](https://github.com/rail-berkeley/d4rl). 

TODO: Link to documentation.
TODO: Prominently mention (and repeat later on) that the repository also provides versions of the datasets with image observations from three cameras.

Some of the datasets were used during the [Real Robot Challenge 2022](https://real-robot-challenge.com).

## Installation

To install the package run with python 3.8 in the root directory of the repository (we recommend doing this in a virtual environment):

```bash
pip install --upgrade pip  # make sure recent version of pip is installed
pip install .
```

## Usage

### Loading the dataset

The datasets are accessible via gym environments which are automatically registered when importing the package. They are automatically downloaded when requested and stored in `~/.trifinger_rl_datasets` as HDF5 files.

The datasets are named following the pattern `trifinger-cube-task-source-quality-v0` where `task` is either `push` or `lift`, `source` is either `sim` or `real` and `quality` can be either `mixed` or `expert`.

By default the observations are loaded as flat arrays. For the simulated datasets the environment can be stepped and visualized. Example usage (also see `demo/load_dataset.py`):

```python
import gymnasium as gym

import trifinger_rl_datasets

env = gym.make(
    "trifinger-cube-push-sim-expert-v0",
    disable_env_checker=True,
    visualization=True,  # enable visualization
)

dataset = env.get_dataset()

print("First observation: ", dataset["observations"][0])
print("First action: ", dataset["actions"][0])
print("First reward: ", dataset["rewards"][0])

obs = env.reset()
done = False
while not done:
    obs, rew, done, info = env.step(env.action_space.sample())
```

Alternatively, the observations can be obtained as nested dictionaries. This simplifies working with the data. As some parts of the observations might be more useful than others, it is also possible to filter the observations when requesting dictionaries (see `demo/load_filtered_dicts.py`):
```python
    # Nested dictionary defines which observations to keep.
    # Everything that is not included or has value False
    # will be dropped.
    obs_to_keep = {
        "robot_observation": {
            "position": True,
            "velocity": True,
            "fingertip_force": False,
        },
        "object_observation": {"keypoints": True},
    }
    env = gym.make(
        args.env_name,
        disable_env_checker=True,
        # filter observations,
        obs_to_keep=obs_to_keep,
    )
```
To transform the observation back to a flat array after filtering, simply set the keyword argument `flatten_obs` to true. Note that the step and reset functions will transform observations in the same manner as the `get_dataset` method to ensure compatibility. A downside of working with observations in the form of dictionaries is that they cause a considerable memory overhead during dataset loading.

All datasets come in two versions: with and without camera observations. The versions with camera observations contain `-image` in their name. Despite PNG image compression they are more than one order of magnitude bigger than the imageless versions. To avoid running out of memory, a part of a dataset can be loaded by specifying a range of timesteps:
```python
env = gym.make(
    "trifinger-cube-push-real-expert-image-v0",
    disable_env_checker=True
)

# load only a subset of obervations, actions and rewards
dataset = env.get_dataset(rng=(1000, 2000))
```
The camera observations corresponding to this range are then returned in `dataset["images"]` with the following dimensions:
```python
n_timesteps, n_cameras, n_channels, height, width = dataset["images"].shape
```
Since the camera frequency is lower than the control frequency, a camera image will repeat over several time steps. To load an array of camera images without this redundancy, the `get_image_data` method can be used:
```python
# images from 3 cameras for each timestep
image_stats = env.get_image_stats()
n_cameras = image_stats["n_cameras"]
images = env.get_image_data(rng=(0, n_cameras * n_camera_frames_to_load)) 
```

### Evaluating a policy in simulation

This package contains an executable module `trifinger_rl_datasets.evaluate_sim`, which
can be used to evaluate a policy in simulation.  As arguments it expects the task
("push" or "lift") and a Python class that implements the policy, following the
`PolicyBase` interface:

    python3 -m trifinger_rl_datasets.evaluate_sim push my_package.MyPolicy

For more options see `--help`.