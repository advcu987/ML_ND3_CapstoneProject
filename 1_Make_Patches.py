import cv2
import imutils
import numpy as np

hwsize = 32

# Original dataset location
orig_dataset_dir = 'Dataset\mitosis'

# This is the directory where the patches will be stored
patches_dataset_dir = 'Dataset\patches'

# loop going through all images and masks
for file in orig_dataset_dir:
    if file.endswith(".tif"):
        print(file)
