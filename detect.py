"""Detection algorithm"""

import os
import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import get_latest_train, get_train_directories, get_latest_detect_suffix, video_inference, image_inference

VIDEO_EXT = [".mp4", ".avi", ".mp3", ".mpeg", ".mpg"]
IMG_EXT = [".bmp",".jpeg",".jpg",".png",".tif",".tiff"]

def parse_args():
    # Setting up parser
    parser = argparse.ArgumentParser(prog="Traffic Light Detector",
                                    description="Detects traffic lights in a given image or video")

    # Setting up arguments for the parser
    # Target directory or file
    parser.add_argument("-t", "--target",
                        help="Target directory or file name")

    # Model to use
    parser.add_argument("-m", "--model",
                        help="Path to the model to use",
                        default="traffic_light_model.pt")

    # Parse the given arguments
    args = parser.parse_args()
    return args


def main(args):
    if args.target is None:
        raise ValueError("'target' argument should not be omitted")

    if not os.path.exists(args.target):
        raise FileNotFoundError(f"Failed to find directory {args.target}")

    if args.model == "latest":
        if not os.path.exists("runs"):
            print("WARN: 'runs' directory not found, using default model")
            if not os.path.exists("traffic_light_model.pt"):
                raise FileNotFoundError("Failed to find the model 'traffic_light_model'. Please reinstall the program")
            MODEL = "traffic_light_model.pt"
        else:
            train_dirs = get_train_directories()
            MODEL = os.path.join("runs", "detect", get_latest_train(train_directories=train_dirs), "weights", "last.pt")
    else:
        MODEL = args.model

    if not os.path.exists(args.target):
        raise FileNotFoundError("No target directory found")

    print(f"""{"#"*30} DETECTION INFO {"#"*30}
    target directory: {args.target}
    model directory: {MODEL}
{"#"*30} END DETECTION INFO {"#"*30}""")

    # Get the model
    model = YOLO(MODEL)

    # If detections directory doesn't exist
    if not os.path.exists("detections"):
        # Create one
        os.mkdir("detections")

    # Get the latest detect suffix
    latest_detect_suffix = get_latest_detect_suffix(os.listdir("detections"))

    # Create the target directory name from the suffix
    new_detect_name = "detect" + str(latest_detect_suffix + 1)

    # Check if the file exists just in case
    if not os.path.exists(os.path.join("detections", new_detect_name)):
        # If not, make it
        os.mkdir(os.path.join("detections", new_detect_name))

    if os.path.isdir(args.target):
        # Get all target files from the directory
        target = []
        for file in os.listdir(args.target):
            target.append(os.path.join("./", args.target, file))
        for elem in target:
            file_name, file_ext = os.path.splitext(elem)
            if file_ext in VIDEO_EXT:
                video_inference(model=model,
                                target_file=elem,
                                latest_predict=new_detect_name,
                                file_name=f"{Path(file_name).stem}{file_ext}")
            elif file_ext in IMG_EXT:
                image_inference(model=model,
                                target_file=elem,
                                latest_predict=new_detect_name)
            else:
                print(f"WARN: Invalid file extension {file_ext} in file {file_name}; skipping")
    else:
        target = args.target
        file_name, file_ext = os.path.splitext(target)
        if file_ext in VIDEO_EXT:
            video_inference(model=model,
                target_file=target,
                latest_predict=new_detect_name,
                file_name=f"{Path(file_name).stem}{file_ext}")
        elif file_ext in IMG_EXT:
            image_inference(model=model,
                            target_file=target,
                            latest_predict=new_detect_name)
        else:
            print(f"WARN: Invalid file extension {file_ext} in file {file_name}; skipping")

    print(f"INFO: Results are saved to 'detections/{new_detect_name}'")

if __name__ == "__main__":
    args = parse_args()
    main(args=args)