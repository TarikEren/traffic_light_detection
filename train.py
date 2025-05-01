""" 
Customised YOLO training script
"""

import os
import argparse

import ultralytics
from ultralytics import YOLO

from utils import get_latest_train, get_train_directories, string_to_bool

# Setting up the argument parser
parser = argparse.ArgumentParser(prog="Traffic Light Model Trainer",
                                 description="Allows the training of the traffic light model")

def parse_args():
    """
    Parses the program arguments
    
    Return
        Arguments namespace
    """
    # Target .yaml file for the YOLO model to use
    parser.add_argument("--yaml",
                        help="Path to the .yaml file",
                        default="./data.yaml")

    # YOLO formatted dataset directory
    parser.add_argument("--dataset",
                        help="Path to the dataset to train with",
                        default="./dataset")

    # Model to train
    parser.add_argument("--model",
                        help="Path to the model to train. 'latest' for the latest iteration",
                        default="./traffic_light_model.pt")

    # Resume training or not
    parser.add_argument("--resume",
                        help="Whether the training should continue from where it left off",
                        type=string_to_bool,
                        const=True,
                        nargs="?",
                        default=False)

    # Epoch count
    parser.add_argument("--epochs",
                        help="For how many epochs the training should proceed",
                        default=100)

    # Image size
    parser.add_argument("--imgsz",
                        help="Image size",
                        default="640")

    # Patience
    parser.add_argument("--patience",
                        help="Count of epochs to halt training after no improvements are visible in the performance metrics",
                        default=100)

    # Workers
    parser.add_argument("--workers",
                        help="Number of workers to use",
                        default=8)

    # Batch size
    parser.add_argument("--batch",
                        help="Batch size",
                        default=16)

    # Starting learning rate
    parser.add_argument("--lr",
                        help="Starting learning rate",
                        default=0.01)
    
    # GPU acceleration
    parser.add_argument("--gpu",
                        help="Whether GPU acceleration should be used",
                        type=string_to_bool,
                        const=True,
                        nargs="?",
                        default=False)

    # Parse the passed arguments
    args = parser.parse_args()

    return args

def main(args):
    # Check if the provided paths exist
    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"Cannot find file {args.yaml}")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Cannot find dataset {args.dataset}")

    # If the latest argument was provided
    if args.model == "latest":
        # Check if runs exists
        if not os.path.exists("runs"):
            # If not, notify the user that it doesn't and fall back to the default model
            print("WARN: 'runs' directory not found, using default model")

            # If the default model wasn't found raise error and kill program
            if not os.path.exists("traffic_light_model.pt"):
                raise FileNotFoundError("Failed to find the model 'traffic_light_model'. Please reinstall the program")
            
            MODEL = "traffic_light_model.pt"
        else:
            # Get the latest model
            train_dirs = get_train_directories()
            MODEL = os.path.join("runs", "detect", get_latest_train(train_directories=train_dirs), "weights", "last.pt")
    else:
        MODEL = args.model

    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            print("WARN: Could not find CUDA, using CPU")
            GPU = False
        else:
            GPU = True
    else:
        GPU = args.gpu

    # Setting constants based on the arguments
    YAML_FILE = args.yaml

    # Get the dataset directory and update YOLO's config accordingly
    DATASET_DIR = args.dataset
    ultralytics.settings.update({"datasets_dir": DATASET_DIR})

    RESUME = args.resume
    EPOCHS = int(args.epochs)
    IMGSZ = int(args.imgsz)
    PATIENCE = int(args.patience)
    WORKERS = int(args.workers)
    BATCH_SIZE = int(args.batch)
    LEARNING_RATE = float(args.lr)

    print(f"""########## TRAINING INFO ##########
    yaml file: {YAML_FILE}
    dataset directory: {DATASET_DIR}
    model directory: {MODEL}
    resume: {RESUME}
    epoch count: {EPOCHS}
    image size: {IMGSZ}
    patience: {PATIENCE}
    worker count: {WORKERS}
    batch size: {BATCH_SIZE}
    starting learning rate: {LEARNING_RATE}
    gpu usage: {GPU}
########## END TRAINING INFO ##########""")

    # Get the model to train using the constant
    model = YOLO(MODEL)

    # Train the model using the pre-set constants
    results = model.train(data=YAML_FILE,
                        epochs=EPOCHS,
                        imgsz=IMGSZ,
                        workers=WORKERS,
                        batch=BATCH_SIZE,
                        patience=PATIENCE,
                        resume=RESUME,
                        lr0=LEARNING_RATE,
                        device="cuda" if GPU else "cpu")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)