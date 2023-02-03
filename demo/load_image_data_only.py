"""Load image data from HDF5 file and display it."""


import argparse

import cv2
import gymnasium as gym
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
        "--n-timesteps",
        type=int,
        default=10,
        help="Number of camera timesteps to load image data for.",
    )
    argparser.add_argument(
        "--h5path",
        type=str,
        default=None,
        help="Path to HDF5 file to load.",
    )
    argparser.add_argument(
        "--do-not-show-images",
        action="store_true",
        help="Do not show images if this is set.",
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
    from time import time
    t0 = time()
    images = env.get_image_data(
        # images from 3 cameras for each timestep
        rng=(0, 3 * args.n_timesteps),
        h5path=args.h5path
    )
    print(f"Loading took {time() - t0:.3f} seconds.")

    # concatenate images from all camera timesteps and cameras and show them
    # ----------------------------------------------------------------------
    if not args.do_not_show_images:
        n_timesteps, n_cameras, n_channels, height, width = images.shape
        output_image = np.zeros(
            (n_cameras * height, n_timesteps * width, n_channels),
            dtype=np.uint8
        )
        # loop over tuples containing images from all cameras at one timestep
        for i, image_tuple in enumerate(images):
            # concatenate images from all cameras along the height axis
            concatenated_image_tuple = np.concatenate(image_tuple, axis=1)
            # change to (height, width, channels) format for cv2
            concatenated_image_tuple = np.transpose(concatenated_image_tuple, (1, 2, 0))
            # copy column of camera images to output image
            output_image[:, i * width:(i + 1) * width, ...] = concatenated_image_tuple
        # convert RGB to BGR for cv2
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        legend = "Each column corresponds to the camera images at one timestep."
        print(legend)
        print("Press any key to close window.")
        cv2.imshow(legend, output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()