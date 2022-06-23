from genericpath import exists
from os import pathsep
import os
from os import path as ops
import tensorflow as tf
import cv2
import random
import pandas as pd
import numpy as np


def read_dataset(file_path, num_samples=4):
    # Function:
    #     Read a file consisting of rows of training dataset path and
    #     seperate them into 3 variables (img_path, bin_path, inst_path)
    # Input:
    #     file_path   : text file consisting of training dataset path
    #     num_samples : number of dataset samples that you want to get from the file 
    # Output:
    #     img_path    : training RGB image paths
    #     bin_path    : training binary segmentation image paths
    #     inst_path   : training instance segmentation image paths

    img_path = []
    bin_path = []
    inst_path = []

    assert exists(file_path), "Training file (txt) does not exist, please make sure the input file is right."

    num_files = sum(1 for line in open(file_path)) #number of records in file (excludes header)
    if num_files < num_samples:
        print('Number of samples is higher than the number of existing files. number of samples is setted to the number of existing files: ', num_files)
        num_samples = num_files
    skip_files = sorted(random.sample(range(1,num_files+1),(num_files-num_samples))) #the 0-indexed header will not be included in the skip list
    text_file = pd.read_csv(file_path, header=None, delim_whitespace=True, skiprows=skip_files, names=['img', 'bin', 'inst'])

    img_path = text_file['img'].values
    bin_path = text_file['bin'].values
    inst_path = text_file['inst'].values

    return img_path, bin_path, inst_path

def img_preprocess(image_path, shape=(512,256)):
    image = cv2.imread(image_path)
#     image = Image.fromarray(image)
#     image = image.resize(shape)
    image = np.array(image, dtype=np.float32)/255.0
    return image
    
def bin_preprocess(bin_path, shape=(512,256)):
    image = cv2.imread(bin_path, 0)
#     image = Image.fromarray(image)
#     image = image.resize(shape)
    label_binary = np.zeros([shape[1],shape[0]], dtype=np.uint8)
    mask = np.where(np.array(image)[:,:] != [0])
    #print(np.unique(np.array(image)))
    label_binary[mask] = 1
    label = np.array(label_binary, dtype=np.uint8)
    label = np.expand_dims(label,-1)
    return label

def inst_preprocess(inst_path, shape=(512,256)):
    image = cv2.imread(inst_path, 0)
    ex = image
#     ex = Image.fromarray(ex)
#     ex = ex.resize(shape)
    label = np.array(ex,dtype=np.float32)
    label = np.expand_dims(label,-1)
    return label

def parse(img_path, bin_path, inst_path):
    img_path = img_path.numpy().decode("utf-8")
    bin_path = bin_path.numpy().decode("utf-8")
    inst_path = inst_path.numpy().decode("utf-8")
    
    assert exists(img_path), f'The dataset file does not exist: {img_path}'
    assert exists(bin_path), f'The dataset file does not exist: {bin_path}'
    assert exists(inst_path), f'The dataset file does not exist: {inst_path}'
    
    imgs = img_preprocess(img_path)
    bins = bin_preprocess(bin_path)
    insts = inst_preprocess(inst_path)

    return imgs, bins, insts

def train_preprocessing(img_path, bin_path, inst_path):
    img, bin, inst = tf.py_function(parse, [img_path, bin_path, inst_path], Tout=[tf.float32, tf.uint8, tf.float32])
    return img, bin, inst
