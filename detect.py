"""Detection algorithm"""

import os
import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import get_latest_train, get_train_directories, get_latest_detect_suffix

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

if os.path.isdir(args.target):
    # Get all target files from the directory
    target = []
    for file in os.listdir(args.target):
        target.append(os.path.join("./", args.target, file))
else:
    target = args.target

print(f"""########## DETECTION INFO ##########
    target directory: {args.target}
    model directory: {MODEL}
########## END DETECTION INFO ##########""")

# Get the model
model = YOLO(MODEL)

# Get the results
results = model(target, stream=True)

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

# Process results list
for result in results:
    # Get the target name
    target_name = str(Path(result.path).stem) + str(Path(result.path).suffix)
    # Get the target path
    target_path = os.path.join("detections", new_detect_name, target_name)
    print(f"Target path: {target_path}")

    boxes = result.boxes                    # Boxes object for bounding box outputs
    masks = result.masks                    # Masks object for segmentation masks outputs
    keypoints = result.keypoints            # Keypoints object for pose outputs
    probs = result.probs                    # Probs object for classification outputs
    obb = result.obb                        # Oriented boxes object for OBB outputs
    result.save(filename=target_path)       # Save to disk

print(f"INFO: Results are saved to 'detections/{new_detect_name}'")