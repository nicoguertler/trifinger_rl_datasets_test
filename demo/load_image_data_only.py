"""Load image data from HDF5 file and display it."""


import argparse

import cv2
import gym
import numpy as np

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-real-expert-image-mini-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--n_timesteps",
        type=int,
        default=10,
        help="Number of timesteps to load image data for.",
    )
    argparser.add_argument(
        "--h5path",
        type=str,
        default=None,
        help="Path to HDF5 file to load.",
    )
    args = argparser.parse_args()

    # create environment
    env = gym.make(
        args.env,
        disable_env_checker=True
    )

    # get information about image data
    image_stats = env.get_image_stats(h5path=args.h5path)
    print("Image dataset:")
    for key, value in image_stats.items():
        print(f"{key}: {value}")

    # load image data
    print(f"Loading {args.n_timesteps} timesteps of image data.")
    images = env.get_image_data(
        # images from 3 cameras for each timestep
        rng=(0, 3 * args.n_timesteps), 
        h5path=args.h5path
    )

    # concatenate images from all camera timesteps and cameras and show them
    n_timesteps, n_cameras, n_channels, height, width = images.shape
    output_image = np.zeros(
        (n_cameras * height, n_timesteps * width, n_channels),
        dtype=np.uint8
    )
    for i, image_tuple in enumerate(images):
        concatenated_image_tuple = np.concatenate(image_tuple, axis=1)
        concatenated_image_tuple = np.transpose(concatenated_image_tuple, (1, 2, 0))
        output_image[:, i*width:(i + 1) * width, ...] = concatenated_image_tuple

    legend = "Each column corresponds to the camera images at one timestep."
    print(legend)
    print("Press any key to close window.")
    cv2.imshow(legend, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()