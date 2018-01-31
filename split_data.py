# Author: Marco Lugo
# Purpose: Split the input folder into training and validation sets
# Usage: python split_data.py validation_ratio
# Example: python split_data.py 0.2

import os
import sys
import random
import shutil

random.seed(404) # Set seed for reproducibility
shape_list = ['circle', 'rectangle']
validation_ratio = float(sys.argv[1])


for subdir, dirs, files in os.walk('./input/images'):
    for fname in files:
        fullname = os.path.join(subdir, fname)

        source_path_img = subdir
        source_path_mask = subdir.replace('images', 'masks')

        if random.random() < validation_ratio:
            dest_path_img = source_path_img.replace('input', 'input/validation')
            dest_path_mask = source_path_mask.replace('input', 'input/validation')
        else:
            dest_path_img = source_path_img.replace('input', 'input/train')
            dest_path_mask = source_path_mask.replace('input', 'input/train')

        # Create directories if missing
        if not os.path.exists(dest_path_img):
            os.makedirs(dest_path_img)
        if not os.path.exists(dest_path_mask):
            os.makedirs(dest_path_mask)

        # Copy files to new directories
        shutil.copy(os.path.join(source_path_img, fname), os.path.join(dest_path_img, fname))
        shutil.copy(os.path.join(source_path_mask, fname), os.path.join(dest_path_mask, fname))
