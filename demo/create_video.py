"""Create video from camera images."""


import argparse

import cv2
import gymnasium as gym
import numpy as np

import trifinger_rl_datasets  # noqa


def create_video(env, output_path, camera_id, timestep_range, zarr_path):
    """Create video from camera images.

    Args:
        dataset (dict):  Dataset to load images from.
        output_path (str):  Output path for video file.
        camera_id (str):  ID of the camera for which to load images.
    """

    image_range = env.convert_timestep_to_image_index(np.array(timestep_range))
    # load relevant part of images in dataset
    images = env.get_image_data(
        # images from 3 cameras for each timestep
        rng=(image_range[0], image_range[1] + 3),
        zarr_path=zarr_path,
        timestep_dimension=True,
    )

    # select only images from the specified camera
    images = images[:, camera_id, ...]

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    video_writer = cv2.VideoWriter(
        output_path, fourcc, fps, (images.shape[-1], images.shape[-2])
    )

    # loop over images
    for image in images:
        # convert to channeel last format for cv2
        img = np.transpose(image, (1, 2, 0))
        # convert RGB to BGR for cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # write image to video
        video_writer.write(img)

    # close video writer
    video_writer.release()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "output_path",
        type=str,
        help="Path to output video file."
    )
    argparser.add_argument(
        "camera_id",
        type=int,
        help="ID of the camera for which to load images.",
    )
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-real-expert-image-mini-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--timestep-range",
        type=int,
        nargs=2,
        default=[0, 750],
        help="Range of timesteps (not camera timesteps) to load image data for.",
    )
    argparser.add_argument(
        "--zarr-path", type=str, default=None, help="Path to Zarr file to load."
    )
    argparser.add_argument(
        "--data-dir", type=str, default=None, help="Path to data directory."
    )
    args = argparser.parse_args()

    env = gym.make(args.env, disable_env_checker=True, data_dir=args.data_dir)
    create_video(env, args.output_path, args.camera_id, args.timestep_range, args.zarr_path)