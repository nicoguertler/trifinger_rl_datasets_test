from copy import deepcopy
import os
from pathlib import Path
from threading import Thread
from typing import Union, Tuple, Dict, Optional, List, Any
import urllib.request

import cv2
import gymnasium as gym
import gymnasium.spaces as spaces
import h5py
import numpy as np
from tqdm import tqdm

from .sim_env import SimTriFingerCubeEnv


class ImageLoader(Thread):
    """Thread for loading and processing images from the dataset.

    This thread is responsible for loading and processing every
    loader_id-th image. Processing includes decoding, reordering
    of pixels and debayering."""

    def __init__(self, loader_id, n_loaders, image_data, image_data_index,
                 unique_images, n_unique_images, n_cameras, offset,
                 reorder_pixels):
        """
        Args:
            loader_id: ID of this loader.  This loader will load every
                loader_id-th image.
            n_loaders: Total number of loaders.
            image_data: Numpy array containing the image data.
            image_data_index: Numpy array containing the indices of the
                start of the images in the image_data array.
            unique_images: Numpy array to which the images are written.
            n_unique_images: Number of unique images to load. If this
                number is not divisible by n_cameras,
                self.unique_images will be padded with zeros.
            n_cameras: Number of cameras.
            offset: Offset compensating for only a part of the image
                data being loaded from the file.
            reorder_pixels: Whether to undo the reordering of the pixels
                which was done during creation of the dataset to improve
                the image compression."""
        super().__init__()
        self.loader_id = loader_id
        self.n_loaders = n_loaders
        self.image_data = image_data
        self.image_data_index = image_data_index
        self.unique_images = unique_images
        self.n_unique_images = n_unique_images
        self.n_cameras = n_cameras
        self.offset = offset
        self.reorder_pixels = reorder_pixels

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

    def _decode_image(self, image: np.ndarray) -> np.ndarray:
        """Decode image from numpy array of type void."""
        # convert numpy array of type V1 to use with cv2 imdecode
        image = np.frombuffer(image, dtype=np.uint8)
        # use cv2 to decode image
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        return image

    def run(self):
        # this thread is responsible for every loader_id-th image
        for i in range(self.loader_id, self.n_unique_images, self.n_loaders):
            timestep, camera = divmod(i, self.n_cameras)
            compressed_image = self.image_data[
                self.image_data_index[i] - self.offset :
                self.image_data_index[i + 1] - self.offset
            ]
            # decode image
            image = self._decode_image(compressed_image)
            if self.reorder_pixels:
                # undo reordering of pixels
                image = self._reorder_pixels(image)
            # debayer image (output channels in RGB order)
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
            # convert to channel first
            self.unique_images[timestep, camera, ...] = np.transpose(image, (2, 0, 1))


class TriFingerDatasetEnv(gym.Env):
    """TriFinger environment which can load an offline RL dataset from a file.

    Similar to D4RL's OfflineEnv but with different data loading and
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
        self._orig_flat_obs_space = spaces.flatten_space(self._orig_obs_space)

        self.name = name
        self.dataset_url = dataset_url
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        self.real_robot = real_robot
        self.obs_to_keep = obs_to_keep
        self.flatten_obs = flatten_obs
        self.scale_obs = scale_obs
        self.set_terminals = set_terminals
        self._local_dataset_path = None

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
            self.observation_space = spaces.flatten_space(self._filtered_obs_space)
            if self.scale_obs:
                self._obs_unscaled_low = self.observation_space.low
                self._obs_unscaled_high = self.observation_space.high
                # scale observations to [-1, 1]
                self.observation_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self.observation_space.shape,
                    dtype=self.observation_space.dtype,
                )
        else:
            self.observation_space = self._filtered_obs_space

    def _download_dataset(self):
        """Download dataset files if not already present."""
        if self._local_dataset_path is None:
            if not isinstance(self.dataset_url, list):
                urls = [self.dataset_url]
            else:
                urls = self.dataset_url
            data_dir = Path("~/.trifinger_rl_datasets").expanduser()
            data_dir.mkdir(exist_ok=True)
            print("Downloading dataset files if not already present.")
            for i, url in tqdm(enumerate(urls)):
                if i == 0:
                    # first URL is the main dataset
                    local_path = data_dir / (self.name + ".hdf5")
                    self._local_dataset_path = local_path
                else:
                    # additional URLs are for the images
                    local_path = data_dir / (self.name + f"_{i - 1}.hdf5")
                if not local_path.exists():
                    print(f'"{url}" to "{local_path}".')
                    urllib.request.urlretrieve(url, local_path)
                    if not local_path.exists():
                        raise IOError(f"Failed to download dataset from {url}.")
        return self._local_dataset_path

    def _filter_dict(self, keys_to_keep, d):
        """Keep only a subset of keys in dict.

        Applied recursively.

        Args:
            keys_to_keep (dict): (Nested) dictionary with values being
                either a dict or a bolean indicating whether to keep
                an item.
            d (dict or gymnasium.spaces.Dict): Dicitionary or Dict space that
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
        if isinstance(d, spaces.Dict):
            filtered_dict = spaces.Dict(spaces=filtered_dict)
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
            obs = spaces.flatten(self._filtered_obs_space, obs)
        if self.scale_obs:
            # scale obs
            obs = self._scale_obs(obs)
        return obs

    def get_obs_indices(self):
        """Get index ranges that correspond to the different observation components.

        Also returns a dictionary containing the shapes of these observation
        components.

        Returns:
            A tuple containing:
            - A dictionary with keys corresponding to the observation components and
            values being tuples of the form (start, end), where start and end are
            the indices at which the observation component starts and ends. The
            nested dictionary structure of the observation is preserved.
            - A dictionary of the same structure but with values being the shapes
            of the observation components."""

        def _construct_dummy_obs(spaces_dict, counter=[0]):
            """Construct dummy observation which has an array repeating
            a different integer as the value of each component."""
            dummy_obs = {}
            for i, (k, v) in enumerate(spaces_dict.items()):
                if isinstance(v, spaces.Dict):
                    dummy_obs[k] = _construct_dummy_obs(v.spaces, counter)
                else:
                    dummy_obs[k] = counter * np.ones(v.shape, dtype=np.int32)
                    counter[0] += 1
            return dummy_obs

        dummy_obs = _construct_dummy_obs(self._orig_obs_space.spaces)
        flat_dummy_obs = spaces.flatten(self._orig_obs_space, dummy_obs)

        def _get_indices_and_shape(dummy_obs, flat_dummy_obs):
            indices = {}
            shape = {}
            for k, v in dummy_obs.items():
                if isinstance(v, dict):
                    indices[k], shape[k] = _get_indices_and_shape(v, flat_dummy_obs)
                else:
                    where = np.where(flat_dummy_obs == v.flatten()[0])[0]
                    indices[k] = (int(where[0]), int(where[-1]) + 1)
                    shape[k] = v.shape
            return indices, shape

        return _get_indices_and_shape(dummy_obs, flat_dummy_obs)

    def get_dataset_stats(self, h5path: Union[str, os.PathLike] = None) -> Dict:
        """Get statistics of dataset such as number of timesteps.

        Args:
            h5path:  Optional path to a HDF5 file containing the dataset, which will be
                used instead of the default.
        Returns:
            The statistics of the dataset as a dictionary with keys:
                - n_timesteps: Number of timesteps in dataset. Corresponds to the
                    number of observations, actions and rewards.
                - obs_size: Size of the observation vector.
                - action_size: Size of the action vector.
        """
        if h5path is None:
            h5path = self._download_dataset()

        with h5py.File(h5path, "r") as dataset_file:
            n_timesteps = dataset_file["observations"].shape[0]
            dataset_stats = {
                "n_timesteps": n_timesteps,
                "obs_size": dataset_file["observations"].shape[1],
                "action_size": dataset_file["actions"].shape[1]
            }
        return dataset_stats

    def get_image_stats(self, h5path: Union[str, os.PathLike] = None) -> Dict:
        """Get statistics of image data in dataset.

        Args:
            h5path:  Optional path to a HDF5 file containing the dataset, which will be
                used instead of the default.
        Returns:
            The statistics of the image data as a dictionary with keys:
                - n_images: Number of images in the dataset.
                - n_cameras: Number of cameras used to capture the images.
                - n_channels: Number of channels in the images.
                - image_shape: Shape of the images in the format (height, width).
                - reorder_pixels: Whether the pixels in the images have been reordered
                    to have the pixels corresponding to one color in the Bayer pattern
                    together in blocks (to improve image compression).
        """
        if h5path is None:
            h5path = self._download_dataset()

        with h5py.File(h5path, "r") as dataset_file:
            image_stats = {
                # have to subtract one because last index contains length of images
                # dataset
                "n_images": dataset_file["image_data_index"].shape[0] - 1,
                "n_cameras": dataset_file["images"].attrs["n_cameras"],
                "n_channels": dataset_file["images"].attrs["n_channels"],
                "image_shape": tuple(dataset_file["images"].attrs["image_shape"]),
                "reorder_pixels": dataset_file["images"].attrs["reorder_pixels"],
            }
        return image_stats

    def get_image_data(
        self,
        rng: Tuple[int, int],
        h5path: Union[str, os.PathLike] = None,
        n_threads: Optional[int] = None
    ) -> np.ndarray:
        """Get image data from dataset.

        Args:
            rng:  Range of images to return. rng=(m,n) means that the images with
                indices m to n-1 are returned.
            h5path:  Optional path to a HDF5 file containing the dataset, which will be
                used instead of the default.
            n_threads: Number of threads to use for processing the images. If None,
                the number of threads is set to the number of CPUs available to the
                process.
        Returns:
            The image data (or a part of it specified by rng) as a numpy array with the
            shape (n_camera_timesteps, n_cameras, n_channels, height, width). The
            channels are ordered as RGB.
        """
        if n_threads is None:
            n_threads = len(os.sched_getaffinity(0))
        if h5path is None:
            h5path = self._download_dataset()
        dataset_file = h5py.File(h5path, "r")

        n_cameras = dataset_file["images"].attrs["n_cameras"]
        n_channels = dataset_file["images"].attrs["n_channels"]
        image_shape = tuple(dataset_file["images"].attrs["image_shape"])
        reorder_pixels = dataset_file["images"].attrs["reorder_pixels"]
        compression = dataset_file["images"].attrs["compression"]
        assert compression == "image", "Only image compression is supported."

        # mapping from image index to start of compressed image data
        # have to load one additional index to obtain size of last image
        image_data_index = dataset_file["image_data_index"][
            slice(rng[0], rng[1] + 1)
        ]
        image_data_range = (image_data_index[0], image_data_index[-1])
        # load only relevant image data
        image_data = dataset_file["images"][slice(*image_data_range)]
        n_unique_images = rng[1] - rng[0]
        n_timesteps = int(np.ceil(n_unique_images / n_cameras))
        unique_images = np.zeros(
            (n_timesteps, n_cameras, n_channels) + image_shape,
            dtype=np.uint8
        )
        offset = image_data_range[0]

        threads = []
        # distribute image loading and processing over multiple threads
        for i in range(n_threads):
            image_loader = ImageLoader(
                loader_id=i,
                n_loaders=n_threads,
                image_data=image_data,
                image_data_index=image_data_index,
                unique_images=unique_images,
                n_unique_images=n_unique_images,
                n_cameras=n_cameras,
                offset=offset,
                reorder_pixels=reorder_pixels
            )
            threads.append(image_loader)
            image_loader.start()
        for thread in threads:
            thread.join()

        return unique_images

    def get_dataset(
        self, h5path: Union[str, os.PathLike] = None, clip: bool = True,
        rng: Optional[Tuple[int, int]] = None,
        n_threads: Optional[int] = None
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
            n_threads: Number of threads to use for processing the images. If None,
                the number of threads is set to the number of CPUs available to the
                process.
        Returns:
            A dictionary with the following items:
                - observations: Either an array or a list of dictionaries
                    containing the observations depending on whether
                    `flatten_obs` is True or False.
                - actions: Array containing the actions.
                - rewards: Array containing the rewards.
                - timeouts: Array containing the timeouts (True only at
                    the end of an episode by default. Always False if
                    `set_terminals` is True).
                - terminals: Array containing the terminals (Always
                    False by default. If `set_terminals` is True, only
                    True at the last timestep of an episode).
                - episode_ends: Array containing the indices of the last
                    timestep of each episode.
                - image_data (only if present in dataset): Array of the
                    shape (n_control_timesteps, n_cameras, n_channels,
                    height, width) containing the image data. The cannels
                    are ordered as RGB.
        """

        # The offline RL dataset is loaded from a HDF5 file which contains
        # the following HDF5 datasets (this is an implementation detail and
        # not necessary to understand for users of the class):
        # - observations: Two-dimensional array of shape
        #     `(n_control_timesteps, n_obs)` containing the observations as
        #     flat vectors of length `n_obs` (except for the camera images
        #     which are stored in image_data if present in the dataset).
        # - actions: Two-dimensional array of shape `(n_control_timesteps,
        #     n_actions)` containing the actions.
        # - rewards: One-dimensional array of length `n_control_timesteps`
        #     containing the rewards.
        # - episode_ends: One-dimensional array of length `n_episodes`
        #     containing the indices of the last control timestep of each
        #     episode.
        # - image_data: One-dimensional array of dtype "V1", i.e., void of
        #     length one byte, which contains the compressed image data. The
        #     images obtained from all cameras at each camera time step are
        #     written one after another to this array. After decompression
        #     the color information is contained in a Bayer pattern. The
        #     images should therefore be debayerd before use. Also note the
        #     information on the reorder_pixels attribute below. The dataset
        #     has the following attributes:
        #     - n_cameras: Number of cameras.
        #     - n_channels: Number of channels per camera image.
        #     - compression: Type of compression used. Only "image" is
        #       supported by this class.
        #     - image_codec: Codec used to compress the image data. Only
        #       "jpeg" and "png" are supported by this class.
        #     - image_shape: Tuple of length 2 containing the height and width
        #       of the images.
        #     - reorder_pixels: If true, the pixels of the Bayer pattern have
        #       been reordered, such that all pixels of a specific colour are
        #       next to each other in one big block (i.e. one block with all
        #       red pixels, one with all blue pixels and one with all green
        #       pixels). This leads to more continuity of the data (compared
        #       to the original Bayer pattern) and thus tends to improve the
        #       performance of standard image compression algorithms (e.g.
        #       PNG). To restore the original image, the pixels need to be
        #       reordered back before debayering.
        # - image_data_index: One-dimensional array of length
        #     `n_camera_timesteps + 1` containing the indices of the start
        #     of the compressed image data for each camera image in
        #     `image_data`. n_cameras consecutive image indices correspond to
        #     the images of all cameras at one camera time step. The last
        #     entry points to len(image_data) and serves as marker for the
        #     end of the last image.
        # - obs_to_image_index: One-dimensional array of length
        #     `n_control_timesteps` containing the index of the camera
        #     image corresponding to each control timestep. This mapping
        #     is necessary because the camera frequency is lower than the
        #     control frequency.

        if h5path is None:
            h5path = self._download_dataset()
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
        for k in self._PRELOAD_KEYS:
            if k == "episode_ends":
                data_dict[k] = dataset_file[k][:]
            else:
                data_dict[k] = dataset_file[k][range_slice]

        n_control_timesteps = data_dict["observations"].shape[0]

        # clip to make sure that there are no outliers in the data
        if clip:
            data_dict["observations"] = data_dict["observations"].clip(
                min=self._orig_flat_obs_space.low,
                max=self._orig_flat_obs_space.high,
                dtype=self._orig_flat_obs_space.dtype,
            )

        if not (self.flatten_obs and self.obs_to_keep is None):
            # unflatten observations, i.e., turn them into dicts again
            unflattened_obs = []
            obs = data_dict["observations"]
            for i in range(obs.shape[0]):
                unflattened_obs.append(
                    spaces.unflatten(self.sim_env.observation_space, obs[i, ...])
                )
            data_dict["observations"] = unflattened_obs

        # timeouts, terminals and info
        episode_ends = data_dict["episode_ends"]
        # Filter episode_ends for entries which are between trans_range[0]
        # and trans_range[1] and make use of episode_ends being sorted.
        start_index = np.searchsorted(episode_ends, rng[0], side="left")
        end_index = np.searchsorted(episode_ends, rng[1], side="left")
        episode_ends = episode_ends[start_index:end_index]
        episode_ends = episode_ends - rng[0]
        data_dict["episode_ends"] = episode_ends
        data_dict["timeouts"] = np.zeros(n_control_timesteps, dtype=bool)
        if not self.set_terminals:
            data_dict["timeouts"][episode_ends] = True
        data_dict["terminals"] = np.zeros(n_control_timesteps, dtype=bool)
        if self.set_terminals:
            data_dict["terminals"][episode_ends] = True
        data_dict["infos"] = [{} for _ in range(n_control_timesteps)]

        # process obs (filtering, flattening, scaling)
        for i in range(n_control_timesteps):
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
                h5path=h5path,
                n_threads=n_threads
            )
            # repeat images to account for control frequency > camera frequency
            images = np.zeros(
                (n_control_timesteps, ) + unique_images.shape[1:], dtype=np.uint8
            )
            for i in range(n_control_timesteps):
                trans_index = (
                    (obs_to_image_index[i] - obs_to_image_index[0]) // n_cameras
                )
                images[i] = unique_images[trans_index]

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
            - observation (dict): agent's observation of the current environment.
            - reward (float): amount of reward returned after previous action.
            - terminated (bool): whether the MDP has reached a terminal state. If true,
                the user needs to call `reset()`.
            - truncated (bool): Whether the truncation condition outside the scope
                of the MDP is satisfied. For this environment this corresponds to a
                timeout. If true, the user needs to call `reset()`.
            - info (dict): info dictionary containing the current time index.
        """
        if self.real_robot:
            raise NotImplementedError(
                "The step method is not available for real-robot data."
            )
        obs, rew, terminated, truncated, info = self.sim_env.step(action, **kwargs)
        # process obs
        processed_obs = self._process_obs(obs)
        return processed_obs, rew, terminated, truncated, info

    def reset(self) -> Tuple[Union[Dict, np.ndarray], Dict]:
        """Reset the environment.

        Returns:
            Tuple of observation and info dictionary.
        """
        if self.real_robot:
            raise NotImplementedError(
                "The reset method is not available for real-robot data."
            )
        obs, info = self.sim_env.reset()
        # process obs
        processed_obs = self._process_obs(obs)
        return processed_obs, info

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

    def reset_fingers(self, reset_wait_time: int = 3000):
        """Moves the fingers to initial position.

        This resets neither the frontend nor the cube. This method is supposed to be
        used for 'soft resets' between episodes in one job.
        """

        if self.real_robot:
            raise NotImplementedError(
                "The reset_fingers method is not available for real-robot data."
            )
        obs, info = self.sim_env.reset_fingers(reset_wait_time)
        processed_obs = self._process_obs(obs)
        return processed_obs, info