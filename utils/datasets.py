# Author: Marco Lugo
# Loading functions for data: reads images and loads them into numpy arrays

import os
import random
import numpy as np
import pandas as pd
from glob import glob
from scipy import misc
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

input_dir = 'input'

def load_from_filelist(filelist):
    imgs = []
    masks = []
    labels = []
    names = []
    for fname in filelist:
        img = misc.imread(fname)
        imgs.append(img)
        mask = misc.imread(fname.replace('images', 'masks'))
        masks.append(mask)
        label = fname.split(os.sep)[-2] # Get class from directory name
        labels.append(label)
        img_name = fname.split(os.sep)[-1].replace('.png', '')
        names.append(int(img_name))
    imgs = np.asarray(imgs)
    masks = np.asarray(masks)
    labels = np.array(labels)
    labels = LabelEncoder().fit_transform(labels)
    labels = to_categorical(labels)
    return imgs, masks, labels, names

def get_bounding_boxes(name_indices, appendix=''): # Passing _n as appendix gets the n_shape bounding boxes
    df_bbox = pd.read_csv(input_dir + '/bounding_boxes'+appendix+'.txt', header=None, prefix='V')
    df_order = pd.DataFrame({'V0':name_indices})
    df_result = pd.merge(df_order, df_bbox, on='V0', how='left')
    np_result = df_result.values
    return np_result[:,1:] # Drop the index (filename), only keep bounding box coordinates

def load_data(): #TODO: implement _n
    train_dir = os.path.join(input_dir, 'train')
    train_dir_img = os.path.join(train_dir, 'images')
    validation_dir = os.path.join(input_dir, 'validation')
    validation_dir_img = os.path.join(validation_dir, 'images')
    train_img_list = glob(os.path.join(train_dir_img, '**/*.png'))
    validation_img_list = glob(os.path.join(validation_dir_img, '**/*.png'))

    X_train, Y_train_masks, Y_train_classes, names_train = load_from_filelist(train_img_list)
    X_val, Y_val_masks, Y_val_classes, names_val = load_from_filelist(validation_img_list)

    Y_train_bbox = get_bounding_boxes(names_train)
    Y_val_bbox = get_bounding_boxes(names_val)

    return X_train, Y_train_masks, Y_train_classes, X_val, Y_val_masks, Y_val_classes, Y_train_bbox, Y_val_bbox
