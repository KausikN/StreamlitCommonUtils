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

def access_video_file_feed_cv2(video_path) -> cv2.VideoCapture:
    '''
    Access a video feed from a file and return the cv2.VideoCapture object

    Args:
        video_path: Path to the video file

    Returns:
        cap: cv2.VideoCapture object
    '''
    return cv2.VideoCapture(video_path)

def access_webcam_feed_cv2() -> cv2.VideoCapture:
    '''
    Access the webcam feed and return the cv2.VideoCapture object

    Returns:
        cap: cv2.VideoCapture object
    '''
    return cv2.VideoCapture(0)

def stream_video_feed_cv2(feed, start_frame=None, end_frame=None, loop=False):
    '''
    Stream frames from a video feed (cv2.VideoCapture) as an iterator
    of frames as numpy arrays.

    Args:
        feed: cv2.VideoCapture object
        start_frame: Optional index of the starting frame to read (inclusive).
                     Defaults to 0.
        end_frame: Optional index of the ending frame to read (exclusive).
                   If None, reads until the end of the video.
        loop: Whether to restart from start_frame after reaching the end
              of the video or end_frame.

    Yields:
        frame: A frame as a numpy array (H, W, 3) in BGR format
    '''
    start_frame = 0 if start_frame is None else start_frame
    frame_idx = start_frame

    if start_frame > 0:
        feed.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        while feed.isOpened():
            if end_frame is not None and frame_idx >= end_frame:
                if not loop:
                    break

                feed.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame
                continue

            ret, frame = feed.read()

            if not ret:
                if not loop:
                    break

                feed.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame
                continue

            yield frame
            frame_idx += 1

    finally:
        feed.release()

def read_video_feed_frames_cv2(feed, start_frame=None, end_frame=None) -> List[np.ndarray]:
    '''
    Read frames from a video feed (cv2.VideoCapture) and return a list of frames as numpy arrays

    Args:
        feed: cv2.VideoCapture object
        start_frame: Optional index of the starting frame to read (inclusive)
        end_frame: Optional index of the ending frame to read (exclusive)

    Returns:
        frames: List of frames (H, W, 3) in BGR format
    '''
    return list(stream_video_feed_cv2(feed, start_frame=start_frame, end_frame=end_frame))

def read_video_file_frames_cv2(video_path, start_frame=None, end_frame=None) -> List[np.ndarray]:
    '''
    Read a video file and return a list of frames as numpy arrays

    Args:
        video_path: Path to the video file
        start_frame: Optional index of the starting frame to read (inclusive)
        end_frame: Optional index of the ending frame to read (exclusive)

    Returns:
        frames: List of frames (H, W, 3) in BGR format
    '''
    feed = cv2.VideoCapture(video_path)
    return read_video_feed_frames_cv2(feed, start_frame=start_frame, end_frame=end_frame)

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