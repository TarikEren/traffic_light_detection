"""Utility functions"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def get_train_suffix(dir: str) -> int:
    """
    Helper function for getting the suffix number of a given train directory

    Args
        (str) dir: Directory name
    Return
        (int) Suffix number
    """
    # Check if the suffix exists
    if dir[5:] == '':
        # If not return 0
        return 0
    # Else return the suffix
    return int(dir[5:])

def get_train_directories() -> list:
    """
    Gets only the directories which have train in their names.
    Assumes that the super-directory of 'runs' exists

    Return
        (list) List of directories
    """
    # Assuming that the runs directory exists
    directories = os.listdir("runs/detect")
    # Empty list for train directories
    train_directories = []

    # For each directory
    for dir in directories:
        # If the name has train in it
        if "train" in dir:
            # Append it to the list
            train_directories.append(dir)

    # Return the list 
    return train_directories

def get_latest_train(train_directories: list) -> str:
    """
    Function for getting the latest train directory

    Args
        (list) train_directories: List of train directories
    Return
        (string) Latest train directory name
    """
    # Set the first train directory as the latest
    latest_train = train_directories[0]
    # Get the latest train's suffix using the helper get_train_suffix
    latest_suffix = get_train_suffix(latest_train)

    # Loop through each directory in the given list
    for dir in train_directories:
        # If the directory's suffix is greater than the latest suffix
        if get_train_suffix(dir) > latest_suffix:
            # Set the latest train as said directory
            latest_train = dir
            # Get it's suffix and set it as the latest suffix
            latest_suffix = get_train_suffix(dir)

    # Return the latest train directory name
    return latest_train

def get_latest_detect_suffix(detect_directories: list) -> str:
    """
    Gets the latest 'detect' folder's suffix number.
    Every detection is saved into its own folder named in the format of
    `detect(numerical suffix)` line detect1 or detect2 and so on
    This function returns the numerical suffix of the newest detect folder.

    Args
        detect_directories (str): List of detection directories
    Return
        (str): Suffix
    """
    if len(detect_directories) == 0:
        return 0
    suffixes = []
    for dir in detect_directories:
        suffixes.append(dir[6:] if dir[6:] != '' else "0")
    suffixes.sort()
    return int(suffixes[len(suffixes) - 1])

def string_to_bool(argparse, string: str):
    """
    Transforms a given string (yes, true, no, false, 1, 0, etc.) into a boolean expression
    Args
        argparse (ArgumentParser): Argument parser in use
        string (str): String to transform
    """
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def video_inference(model: YOLO, target_file: str, latest_predict: str, file_name: str):
    """
    Performs inference on a given video file
    Args
        model (YOLO): Path to the YOLO model
        target_file (str): Path to the target file
        latest_predict (str): Name of the latest 'predict' directory
        file_name (str): Saved file name
    """
    # Set environment variables to force FFmpeg
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'

    # Open the video file
    cap = cv2.VideoCapture(target_file)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.join("detections", latest_predict)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create video writer with H264 codec
    output_path = os.path.join(output_dir, file_name)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run prediction
        results = model(frame)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames")

    # Release everything
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")

def image_inference(model: YOLO, target_file: str,  latest_predict: str):
    """
    Performs inference on a given image
        model (YOLO): Path to the YOLO model
        target_file (str): Path to the target file
        latest_predict (str): Name of the latest 'predict' directory
    """
    # Get the results
    results = model(target_file, stream=True)

    # Process results list
    for result in results:
        # Get the target name
        target_name = str(Path(result.path).stem) + str(Path(result.path).suffix)

        # Get the target path
        target_path = os.path.join("detections", latest_predict, target_name)
        print(f"Target path: {target_path}")

        # Save to disk
        result.save(filename=target_path)