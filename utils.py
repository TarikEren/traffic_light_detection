"""Utility functions"""

import os

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
        suffixes.append(dir[7:] if dir[7:] != '' else "0")
    suffixes.sort()
    return int(suffixes[len(suffixes) - 1])