# Imports
import os
import cv2
import imageio
import subprocess
import numpy as np
from typing import List
from pathlib import Path
from matplotlib.animation import PillowWriter
from moviepy import ImageClip, concatenate_videoclips

# Main Functions
def save_images_as_video(
    images: List[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> None:
    '''
    Save a list of numpy image arrays as an MP4 video

    Args:
        images: List of images (H, W, 3) in BGR or RGB format
        output_path: Path to output video file
        fps: Frames per second

    Returns:
        None
    '''
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
    '''
    Save a list of numpy image arrays as a GIF

    Args:
        images: List of images (H, W, 3) in BGR or RGB format
        output_path: Path to output GIF file
        fps: Frames per second

    Returns:
        None
    '''
    duration = 1.0 / fps
    imageio.mimsave(output_path, images, duration=duration)

def save_matplotlib_animation_as_gif(
    anim,
    output_path: str,
    fps: int = 10,
) -> None:
    '''
    Save a Matplotlib animation as a GIF

    Args:
        anim: Matplotlib FuncAnimation object
        output_path: Path to output GIF file
        fps: Frames per second

    Returns:
        None
    '''
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, fps=fps)

def save_images_as_video_moviepy(frames, save_path, fps=24.0) -> None:
    '''
    Save a list of images as a GIF or Video using MoviePy

    Args:
        frames (list): List of images (H, W, 3) in BGR or RGB format
        save_path (str): Path to save the video file
        fps (float): Frames per second for the video

    Returns:
        None
    '''
    # Init
    frame_duration = 1.0 / fps
    FRAMES = []
    # Create Image Clips
    for i in range(len(frames)):
        frame_clip = ImageClip(frames[i]).with_duration(frame_duration)
        FRAMES.append(frame_clip)
    # Concatenate
    VIDEO = concatenate_videoclips(FRAMES, method="chain")
    # Write Video
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    VIDEO.write_videofile(save_path, fps=fps)

def read_video_cv2(video_path, start_frame=None, end_frame=None) -> List[np.ndarray]:
    '''
    Read a video file and return a list of frames as numpy arrays

    Args:
        video_path: Path to the video file
        start_frame: Optional index of the starting frame to read (inclusive)
        end_frame: Optional index of the ending frame to read (exclusive)

    Returns:
        frames: List of frames (H, W, 3) in BGR format
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if (start_frame is not None and frame_idx < start_frame) or (end_frame is not None and frame_idx >= end_frame):
            frame_idx += 1
            continue
        frames.append(frame)
        frame_idx += 1

def access_webcam_feed_cv2() -> cv2.VideoCapture:
    '''
    Access the webcam feed and return a list of frames as numpy arrays

    Returns:
        frames: List of frames (H, W, 3) in BGR format
    '''
    return cv2.VideoCapture(0)

def reencode_video_ffmpeg(input_path, output_path) -> None:
    '''
    Re-encode a video file using FFmpeg to ensure compatibility

    Args:
        input_path: Path to the input video file
        output_path: Path to the output video file

    Returns:
        None
    '''
    if os.path.exists(output_path): os.remove(output_path)

    COMMAND_VIDEO_CONVERT = "ffmpeg -i \"{path_in}\" -vcodec libx264 \"{path_out}\""
    convert_cmd = COMMAND_VIDEO_CONVERT.format(path_in=input_path, path_out=output_path)
    print("Running Conversion Command...")
    print(convert_cmd + "\n")
    ConvertOutput = subprocess.getoutput(convert_cmd)
    print("Conversion Output: \n" + ConvertOutput + "\n")