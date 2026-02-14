# Imports
import cv2
import imageio
from pathlib import Path
from typing import List
import numpy as np

# Main Functions
def save_images_as_video(
    images: List[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> None:
    """
    Save a list of numpy image arrays as an MP4 video.

    Args:
        images: List of images (H, W, 3) in BGR or RGB format.
        output_path: Path to output video file.
        fps: Frames per second.
    """
    if not images:
        raise ValueError("No images provided.")

    height, width, _ = images[0].shape
    output_path = str(Path(output_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        video.write(img)

    video.release()


def save_images_as_gif(
    images: List[np.ndarray],
    output_path: str,
    fps: int = 10,
) -> None:
    """
    Save a list of numpy image arrays as a GIF.
    """
    duration = 1 / fps
    imageio.mimsave(output_path, images, duration=duration)
