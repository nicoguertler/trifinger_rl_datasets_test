from copy import deepcopy
import os
from typing import Union, Tuple, Dict, Optional, List, Any
import urllib.request

import cv2
import gym
import gym.spaces
import h5py
import imagecodecs as ic
import numpy as np
from tqdm import tqdm

from .sim_env import SimTriFingerCubeEnv


def download_dataset(url, name):
    data_dir = os.path.expanduser("~/.trifinger_rl_datasets")
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, name + ".hdf5")
    if not os.path.exists(local_path):
        print(f'Downloading dataset "{url}" to "{local_path}".')
        urllib.request.urlretrieve(url, local_path)
        if not os.path.exists(local_path):
            raise IOError(f"Failed to download dataset from {url}.")
    return local_path


class TriFingerDatasetEnv(gym.Env):
    """TriFinger environment which can load an offline RL dataset from a file.

    Similar to D4RL's OfflineEnv but with slightly different data loading and
    options for customization of observation space."""

    _PRELOAD_KEYS = ["observations", "actions", "rewards", "episode_ends"]

    def __init__(
        self,
        name,
        dataset_url,
        ref_max_score,
        ref_min_score,
        trifinger_kwargs,
        real_robot=False,
        visualization=None,
        obs_to_keep=None,
        flatten_obs=True,
        scale_obs=False,
        set_terminals=False,
        **kwargs,
    ):
        """
        Args:
            name (str): Name of the dataset.
            dataset_url (str): URL pointing to the dataset.
            ref_max_score (float): Maximum score (for score normalization)
            ref_min_score (float): Minimum score (for score normalization)
            trifinger_kwargs (dict): Keyword arguments for underlying
                SimTriFingerCubeEnv environment.
            real_robot (bool): Whether the data was collected on real
                robots.
            visualization (bool): Enables rendering for simulated
                environment.
            obs_to_keep (dict): Dictionary with the same structure as
                the observation of SimTriFingerCubeEnv. The boolean
                value of each item indicates whether it should be
                included in the observation. If None, the
                SimTriFingerCubeEnv is used.
            flatten_obs (bool): Whether to flatten the observation. Can
                be combined with obs_to_keep.
            scale_obs (bool): Whether to scale all components of the
                observation to interval [-1, 1]. Only implemented
                for flattend observations.
        """
        super().__init__(**kwargs)
        t_kwargs = deepcopy(trifinger_kwargs)
        if visualization is not None:
            t_kwargs["visualization"] = visualization
        # underlying simulated TriFinger environment
        self.sim_env = SimTriFingerCubeEnv(**t_kwargs)
        self._orig_obs_space = self.sim_env.observation_space

        self.name = name
        self.dataset_url = dataset_url
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        self.real_robot = real_robot
        self.obs_to_keep = obs_to_keep
        self.flatten_obs = flatten_obs
        self.scale_obs = scale_obs
        self.set_terminals = set_terminals

        if scale_obs and not flatten_obs:
            raise NotImplementedError(
                "Scaling of observations only "
                "implemented for flattened observations, i.e., for "
                "flatten_obs=True."
            )

        # action space
        self.action_space = self.sim_env.action_space

        # observation space
        if self.obs_to_keep is not None:
            # construct filtered observation space
            self._filtered_obs_space = self._filter_dict(
                keys_to_keep=self.obs_to_keep, d=self.sim_env.observation_space
            )
        else:
            self._filtered_obs_space = self.sim_env.observation_space
        if self.flatten_obs:
            # flat obs space
            self.observation_space = gym.spaces.flatten_space(self._filtered_obs_space)
            if self.scale_obs:
                self._obs_unscaled_low = self.observation_space.low
                self._obs_unscaled_high = self.observation_space.high
                # scale observations to [-1, 1]
                self.observation_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self.observation_space.shape,
                    dtype=self.observation_space.dtype,
                )
        else:
            self.observation_space = self._filtered_obs_space

    def _filter_dict(self, keys_to_keep, d):
        """Keep only a subset of keys in dict.

        Applied recursively.

        Args:
            keys_to_keep (dict): (Nested) dictionary with values being
                either a dict or a bolean indicating whether to keep
                an item.
            d (dict or gym.spaces.Dict): Dicitionary or Dict space that
                is to be filtered."""

        filtered_dict = {}
        for k, v in keys_to_keep.items():
            if isinstance(v, dict):
                subspace = self._filter_dict(v, d[k])
                filtered_dict[k] = subspace
            elif isinstance(v, bool) and v:
                filtered_dict[k] = d[k]
            elif not isinstance(v, bool):
                raise TypeError(
                    "Expected boolean to indicate whether item "
                    "in observation space is to be kept."
                )
        if isinstance(d, gym.spaces.Dict):
            filtered_dict = gym.spaces.Dict(spaces=filtered_dict)
        return filtered_dict

    def _scale_obs(self, obs: np.ndarray) -> np.ndarray:
        """Scale observation components to [-1, 1]."""

        interval = self._obs_unscaled_high.high - self._obs_unscaled_low.low
        a = (obs - self._obs_unscaled_low.low) / interval
        return a * 2.0 - 1.0

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Process obs according to params."""

        if self.obs_to_keep is not None:
            # filter obs
            if self.obs_to_keep is not None:
                obs = self._filter_dict(self.obs_to_keep, obs)
        if self.flatten_obs and isinstance(obs, dict):
            # flatten obs
            obs = gym.spaces.flatten(self._filtered_obs_space, obs)
        if self.scale_obs:
            # scale obs
            obs = self._scale_obs(obs)
        return obs

    def _decode_image(self, image: np.ndarray) -> np.ndarray:
        """Decode image from numpy array of type void."""
        image = bytes(image)
        image = ic.png_decode(image)
        return image

    def _reorder_pixels(self, img: np.ndarray) -> np.ndarray:
        """Undo reordering of Bayer pattern."""
        new = np.empty_like(img)
        a = img.shape[0] // 2
        b = img.shape[1] // 2

        red = img[0:a, 0:b]
        blue = img[a:, 0:b]
        green1 = img[0:a, b:]
        green2 = img[a:, b:]

        new[0::2, 0::2] = red
        new[1::2, 1::2] = blue
        new[0::2, 1::2] = green1
        new[1::2, 0::2] = green2

        return new

    def get_image_stats(self, h5path: Union[str, os.PathLike] = None) -> Dict:
        """Get image statistics from dataset.
        
        Args:
            h5path:  Optional path to a HDF5 file containing the dataset, which will be
                used instead of the default.
        Returns:
            The image statistics.
        """
        if h5path is None:
            h5path = download_dataset(self.dataset_url, self.name)

        with h5py.File(h5path, "r") as dataset_file:
            image_stats = {
                # have to subtract one because last index contains length of images dataset
                "n_images": dataset_file["image_data_indices"].shape[0] - 1,
                "n_cameras": dataset_file["images"].attrs["n_cameras"],
                "n_channels": dataset_file["images"].attrs["n_channels"],
                "image_shape": tuple(dataset_file["images"].attrs["image_shape"]),
                "reorder_pixels": dataset_file["images"].attrs["reorder_pixels"],
            }
        return image_stats

    def get_image_data(self, rng:Tuple[int, int], h5path: Union[str, os.PathLike] = None) -> Dict:
        """Get image data from dataset.
        
        Args:
            rng:  Range of images to return. rng=(m,n) means that the images with indices
                m to n-1 are returned.
            h5path:  Optional path to a HDF5 file containing the dataset, which will be
                used instead of the default.
        Returns:
            The image data (or a part of it specified by rng).
        """
        if h5path is None:
            h5path = download_dataset(self.dataset_url, self.name)
        dataset_file = h5py.File(h5path, "r")

        n_cameras = dataset_file["images"].attrs["n_cameras"]
        n_channels = dataset_file["images"].attrs["n_channels"]
        image_shape = tuple(dataset_file["images"].attrs["image_shape"])
        reorder_pixels = dataset_file["images"].attrs["reorder_pixels"]

        # mapping from image index to start of compressed image data
        # have to load one additional index to obtain size of last image
        image_data_indices = dataset_file["image_data_indices"][slice(rng[0], rng[1] + 1)]
        image_data_range = (image_data_indices[0], image_data_indices[-1])
        # load only relevant image data
        image_data = dataset_file["images"][slice(*image_data_range)]
        n_unique_images = rng[1] - rng[0]
        n_timesteps = int(np.ceil(n_unique_images / n_cameras))
        unique_images = np.zeros(
            (n_timesteps, n_cameras, n_channels) + image_shape,
            dtype=np.uint8
        )
        offset = image_data_range[0]
        # TODO: Should we parallelize this?
        for i in range(n_unique_images):
            timestep = i // n_cameras
            camera = i % n_cameras
            compressed_image = image_data[image_data_indices[i] - offset: image_data_indices[i+1] - offset]
            # decode image
            image = self._decode_image(compressed_image)
            if reorder_pixels:
                # undo reordering of pixels
                image = self._reorder_pixels(image)
            # debayer image
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
            # convert to channel first
            unique_images[timestep, camera, ...] = np.transpose(image, (2, 0, 1))

        return unique_images

    def get_dataset(
        self, h5path: Union[str, os.PathLike] = None, clip: bool = True,
        rng: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Get the dataset.

        When called for the first time, the dataset is automatically downloaded and
        saved to ``~/.trifinger_rl_datasets``.

        Args:
            h5path:  Optional path to a HDF5 file containing the dataset, which will be
                used instead of the default.
            clip:  If True, observations are clipped to be within the environment's
                observation space.
            rng:  Optional range to return. rng=(m,n) means that observations, actions
                and rewards m to n-1 are returned. If not specified, the entire
                dataset is returned.
        Returns:
            The dataset (or a part of it specified by rng).
        """
        if h5path is None:
            h5path = download_dataset(self.dataset_url, self.name)
        dataset_file = h5py.File(h5path, "r")

        # turn range into slice
        n_avail_transitions = dataset_file["observations"].shape[0]
        if rng is None:
            rng = (None, None)
        rng = (
            0 if rng[0] is None else rng[0],
            n_avail_transitions if rng[1] is None else rng[1]

        )
        range_slice = slice(*rng)

        data_dict = {}
        for k in tqdm(self._PRELOAD_KEYS, desc="Loading datafile"):
            if k == "episode_ends":
                data_dict[k] = dataset_file[k][:]
            else:
                data_dict[k] = dataset_file[k][range_slice]

        n_transitions = data_dict["observations"].shape[0]

        # clip to make sure that there are no outliers in the data
        if clip:
            orig_flat_obs_space = gym.spaces.flatten_space(self._orig_obs_space)
            data_dict["observations"] = data_dict["observations"].clip(
                min=orig_flat_obs_space.low,
                max=orig_flat_obs_space.high,
                dtype=orig_flat_obs_space.dtype,
            )

        if not (self.flatten_obs and self.obs_to_keep is None):
            # unflatten observations, i.e., turn them into dicts again
            unflattened_obs = []
            obs = data_dict["observations"]
            for i in range(obs.shape[0]):
                unflattened_obs.append(
                    gym.spaces.unflatten(self.sim_env.observation_space, obs[i, ...])
                )
            data_dict["observations"] = unflattened_obs

        # timeouts, terminals and info
        episode_ends = data_dict["episode_ends"]
        if rng is not None:
            # Filter episode_ends for entries which are between trans_range[0]
            # and trans_range[1] and make use of episode_ends being sorted.
            start_index = np.searchsorted(episode_ends, rng[0], side="left")
            end_index = np.searchsorted(episode_ends, rng[1], side="left")
            episode_ends = episode_ends[start_index:end_index]
            episode_ends = episode_ends - rng[0]
            data_dict["episode_ends"] = episode_ends
        data_dict["timeouts"] = np.zeros(n_transitions, dtype=bool)
        if not self.set_terminals:
            data_dict["timeouts"][episode_ends] = True
        data_dict["terminals"] = np.zeros(n_transitions, dtype=bool)
        if self.set_terminals:
            data_dict["terminals"][episode_ends] = True
        data_dict["infos"] = [{} for _ in range(n_transitions)]

        # process obs (filtering, flattening, scaling)
        for i in range(n_transitions):
            data_dict["observations"][i] = self._process_obs(
                data_dict["observations"][i]
            )
        # turn observations into array if obs are flattened
        if self.flatten_obs:
            data_dict["observations"] = np.array(
                data_dict["observations"], dtype=self.observation_space.dtype
            )

        if "images" in dataset_file.keys():
            # mapping from observation index to image index
            # (necessary since the camera frequency < control frequency)
            obs_to_image_index = dataset_file["obs_to_image_index"][range_slice]
            n_cameras = dataset_file["images"].attrs["n_cameras"]
            image_index_range = (
                obs_to_image_index[0],
                # add n_cameras to include last images as well
                obs_to_image_index[-1] + n_cameras
            )
            # load images
            unique_images = self.get_image_data(
                rng=image_index_range,
                h5path=h5path
            )
            # repeat images to account for control frequency > camera frequency
            images = np.zeros((n_transitions, ) + unique_images.shape[1:], dtype=np.uint8)
            for i in range(n_transitions):
                trans_index = (obs_to_image_index[i] - obs_to_image_index[0]) // n_cameras
                images[i] = unique_images[trans_index]

            data_dict["obs_to_image_index"] = obs_to_image_index # TODO: Should we drop this?
            data_dict["unique_images"] = unique_images # TODO: Should we drop this?
            data_dict["images"] = images

        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        raise NotImplementedError()

    def compute_reward(
        self, achieved_goal: dict, desired_goal: dict, info: dict
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current pose of the object.
            desired_goal: Goal pose of the object.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal.
        """
        return self.sim_env.compute_reward(achieved_goal, desired_goal, info)

    def step(
        self, action: np.ndarray, **kwargs
    ) -> Tuple[Union[Dict, np.ndarray], float, bool, Dict]:
        """Execute one step.

        Args:
            action: Array of 9 torque commands, one for each robot joint.

        Returns:
            A tuple with

            - observation (dict or array): observation of the current environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.real_robot:
            raise NotImplementedError(
                "The step method is not available for real-robot data."
            )
        obs, rew, done, info = self.sim_env.step(action, **kwargs)
        # process obs
        processed_obs = self._process_obs(obs)
        return processed_obs, rew, done, info

    def reset(
        self, return_info: bool = False
    ) -> Union[Union[Dict, np.ndarray], Tuple[Union[Dict, np.ndarray], Dict]]:
        """Reset the environment.

        Args:
            return_info:  If true, an "info" dictionary is returned in addition to the
                observation.

        Returns:
            If return_info is false: Observation of the initial environment state.
            If return_info is true: Tuple of observation and info dictionary.
        """
        if self.real_robot:
            raise NotImplementedError(
                "The reset method is not available for real-robot data."
            )
        rvals = self.sim_env.reset(return_info)
        if return_info:
            obs, info = rvals
        else:
            obs = rvals
        # process obs
        processed_obs = self._process_obs(obs)
        if return_info:
            return processed_obs, info
        else:
            return processed_obs

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed of the environment."""
        return self.sim_env.seed(seed)

    def render(self, mode: str = "human"):
        """Does not do anything for this environment."""
        if self.real_robot:
            raise NotImplementedError(
                "The render method is not available for real-robot data."
            )
        self.sim_env.render(mode)

    def reset_fingers(self, reset_wait_time: int = 3000, return_info: bool = False):
        """Moves the fingers to initial position.

        This resets neither the frontend nor the cube. This method is supposed to be
        used for 'soft resets' between episodes in one job.
        """

        if self.real_robot:
            raise NotImplementedError(
                "The reset_fingers method is not available for real-robot data."
            )
        rvals = self.sim_env.reset_fingers(reset_wait_time, return_info)
        if return_info:
            obs, info = rvals
        else:
            obs = rvals
        # process obs
        processed_obs = self._process_obs(obs)
        if return_info:
            return processed_obs, info
        else:
            return processed_obs
