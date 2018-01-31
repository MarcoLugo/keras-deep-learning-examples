# Author: Marco Lugo
# Purpose: Generate random images for training (convolutional) neural networks
# Usage: python create_images.py img_size n_imgs n_shapes
# Example: python create_images.py 128 500 1

import numpy as np
import random
import cv2
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

random.seed(404) # Set seed for reproducibility
shape_list = ['circle', 'rectangle']
base_dir_img = './input/images/'
base_dir_masks = './input/masks/'
base_dir_img2 = './input/images_n/'
base_dir_masks2 = './input/masks_n/'

# Arguments passed
img_size = int(sys.argv[1])
n_imgs = int(sys.argv[2])
n_shapes = int(sys.argv[3])


def add_circle(img, mask):
    img_size = img.shape[0] # Assumption: b = h

    shape_color = (random.randint(img[:,:,0].min()+1,255), # color will be the minimum value for the channel as it always darker than that of the shape's
                   random.randint(img[:,:,1].min()+1,255),
                   random.randint(img[:,:,2].min()+1,255))
    shape_radius = random.randint(int(img_size/10), int(img_size/3))
    shape_center = (random.randint(0,img_size), random.randint(0,img_size))

    cv2.circle(img, center=shape_center, radius=shape_radius, color=shape_color, thickness=-1)
    cv2.circle(mask, center=shape_center, radius=shape_radius, color=255, thickness=-1)
    bounding_box = (shape_center[0]-shape_radius, shape_center[1]-shape_radius, shape_radius*2, shape_radius*2) # x1,y1,w,h
    return img, mask, bounding_box

def add_rectangle(img, mask):
    img_size = img.shape[0] # Assumption: b = h
    shape_color = (random.randint(img[:,:,0].min()+1,255),
                   random.randint(img[:,:,1].min()+1,255),
                   random.randint(img[:,:,2].min()+1,255))
    shape_radius = random.randint(int(img_size/10), int(img_size/3))
    shape_x1y1 = (random.randint(0,img_size), random.randint(0,img_size))
    shape_x2y2 = (random.randint(0,img_size), random.randint(0,img_size))

    cv2.rectangle(img, shape_x1y1, shape_x2y2, color=shape_color, thickness=-1)
    cv2.rectangle(mask, shape_x1y1, shape_x2y2, color=255, thickness=-1)
    bounding_box = (min(shape_x1y1[0], shape_x2y2[0]), min(shape_x1y1[1], shape_x2y2[1]), abs(shape_x2y2[0]-shape_x1y1[0]), abs(shape_x2y2[1]-shape_x1y1[1]))
    return img, mask, bounding_box

def create_random_img(img_size=64, noise_level=0.1, n_shapes=1, color_threshold=127):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8) # Unsigned integer (0 to 255)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Random color for background
    img[:,:,0] = random.randint(0, color_threshold)
    img[:,:,1] = random.randint(0, color_threshold)
    img[:,:,2] = random.randint(0, color_threshold)

    label = random.choice(shape_list)
    bounding_box = ()
    for i in range(n_shapes): # For now we stick with the same shape every time
        if label == 'circle':
            img, mask, bounding_box_tmp = add_circle(img, mask)
        elif label == 'rectangle':
            img, mask, bounding_box_tmp = add_rectangle(img, mask)
        bounding_box += bounding_box_tmp # concatenate bounding boxes

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
    img_dir2 = base_dir_img2 + shape
    mask_dir2 = base_dir_masks2 + shape
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(img_dir2):
        os.makedirs(img_dir2)
    if not os.path.exists(mask_dir2):
        os.makedirs(mask_dir2)

# Create images
# Single shape per image
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

# n shapes per image where n > 1
for i in range(n_imgs):
    img2, mask2, label2, bounding_box2 = create_random_img(img_size=img_size, n_shapes=n_shapes)
    image_file = Image.fromarray(img2)
    image_file.save(base_dir_img2 + label2 + '/' + str(i) + '.png')
    mask_file = Image.fromarray(mask2)
    mask_file.save(base_dir_masks2 + label2 + '/' + str(i) + '.png')
    bounding_box_string = str(i) + ',' + str(bounding_box2) + '\n'
    bounding_box_string = bounding_box_string.replace('(', '')
    bounding_box_string = bounding_box_string.replace(')', '')
    with open('./input/bounding_boxes_n.txt', 'a') as bounding_box_file:
        bounding_box_file.write(bounding_box_string)
