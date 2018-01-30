# Author: Marco Lugo
# Purpose: Generate random images for training (convolutional) neural networks
# Usage: python create_images.py img_size n_imgs
# Example: python create_images.py 128 500

import numpy as np
import random
import cv2
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

random.seed(404) # Set seed for reproducibility
shape_list = ['circle', 'rectangle']
visualize = False
base_dir_img = './input/images/'
base_dir_masks = './input/masks/'

# Arguments passed
img_size = int(sys.argv[1])
n_imgs = int(sys.argv[2])


def add_circle(img, mask):
    img_size = img.shape[0] # Assumption: b = h
    shape_color = (random.randint(img[0,0,0]+1,255),
                   random.randint(img[0,0,1]+1,255),
                   random.randint(img[0,0,2]+1,255))
    shape_radius = random.randint(int(img_size/10), int(img_size/3))
    shape_center = (random.randint(0,img_size), random.randint(0,img_size))

    cv2.circle(img, center=shape_center, radius=shape_radius, color=shape_color, thickness=-1)
    cv2.circle(mask, center=shape_center, radius=shape_radius, color=255, thickness=-1)
    bounding_box = (shape_center[0]-shape_radius, shape_center[1]-shape_radius, shape_center[0]+shape_radius, shape_center[1]+shape_radius) # x1,y1,x2,y2
    return img, mask, bounding_box

def add_rectangle(img, mask):
    img_size = img.shape[0] # Assumption: b = h
    shape_color = (random.randint(img[0,0,0]+1,255),
                   random.randint(img[0,0,1]+1,255),
                   random.randint(img[0,0,2]+1,255))
    shape_radius = random.randint(int(img_size/10), int(img_size/3))
    shape_x1y1 = (random.randint(0,img_size), random.randint(0,img_size))
    shape_x2y2 = (random.randint(0,img_size), random.randint(0,img_size))

    cv2.rectangle(img, shape_x1y1, shape_x2y2, color=shape_color, thickness=-1)
    cv2.rectangle(mask, shape_x1y1, shape_x2y2, color=255, thickness=-1)
    bounding_box = shape_x1y1 + shape_x2y2
    return img, mask, bounding_box

def create_random_img(img_size=64, noise_level=0.1, color_threshold=127):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8) # Unsigned integer (0 to 255)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Random color for background
    img[:,:,0] = random.randint(0, color_threshold)
    img[:,:,1] = random.randint(0, color_threshold)
    img[:,:,2] = random.randint(0, color_threshold)

    label = random.choice(shape_list)
    if label == 'circle':
        img, mask, bounding_box = add_circle(img, mask)
    elif label == 'rectangle':
        img, mask, bounding_box = add_rectangle(img, mask)

    # Add white noise
    density = random.uniform(0, noise_level)
    for i in range(img_size):
        for j in range(img_size):
            if random.random() < density:
                img[i,j,0] = random.randint(0, 255)
                img[i,j,1] = random.randint(0, 255)
                img[i,j,2] = random.randint(0, 255)

    return img, mask, label, bounding_box

def show_img(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()


# Create directories
for shape in shape_list:
    img_dir = base_dir_img + shape
    mask_dir = base_dir_masks + shape
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

# Create images
for i in range(n_imgs):
    img1, mask1, label1, bounding_box1 = create_random_img(img_size=img_size)
    image_file = Image.fromarray(img1)
    image_file.save(base_dir_img + label1 + '/' + str(i) + '.png')
    mask_file = Image.fromarray(mask1)
    mask_file.save(base_dir_masks + label1 + '/' + str(i) + '.png')
    bounding_box_string = str(i) + ',' + str(bounding_box1) + '\n'
    bounding_box_string = bounding_box_string.replace('(', '')
    bounding_box_string = bounding_box_string.replace(')', '')
    with open('./input/bounding_boxes.txt', 'a') as bounding_box_file:
        bounding_box_file.write(bounding_box_string)

    if visualize == True:
        print(label1 + ' : ' + str(bounding_box1))
        show_img(img1)
        show_img(mask1)
