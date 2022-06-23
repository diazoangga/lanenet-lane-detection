## IMPORT ALL THE LIBRARIES
import os
import os.path as ops
from os.path import exists
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from focal_loss import SparseCategoricalFocalLoss

# import all the predefined functions
from config_utils import Config
from dataset_utils import *
from bisenetv2 import *
from loss_function import instance_loss, CalculateIOU

CFG = Config(config_path='./lanenet.yml')
testing_path = CFG.DATASET.TRAIN_FILE_LIST
img_size = CFG.DATASET.IMAGE_SIZE
# num_samples = CFG.DATASET.MAX_NUM_SAMPLES
# val_ratio = CFG.DATASET.VAL_RATIO
# num_epochs = CFG.TRAIN.EPOCH_NUMS
batch_size = CFG.TRAIN.BATCH_SIZE
# val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
# learning_rate = CFG.SOLVER.LR
loss_weights = CFG.TRAIN.LOSS_WEIGHTS
# save_path = CFG.TRAIN.MODEL_SAVE_DIR
model_path = CFG.MODEL.WEIGHT_PATH
# warmup_epochs = CFG.TRAIN.WARMUP_EPOCHS


## IMPORT ALL THE DATASETS
print('Importing the datasets with the following parameters...')
print('   Testing path                    :', testing_path)

img_test, bin_test, inst_test = read_dataset(testing_path, num_samples=10)

test_ds = tf.data.Dataset.from_tensor_slices((img_test, bin_test, inst_test))

print(f'\nImporting the datasets with {len(list(test_ds))} testing data is completed')


## CHECK IF THE PREPROCESSING FUNCTION WORKS PROPERLY
test_dataset = test_ds.map(train_preprocessing).cache().shuffle(100).batch(batch_size, drop_remainder=True)

test_dataset = iter(test_dataset)



## BUILD THE MODEL
tf.random.set_seed(40)
model = BiseNetV2().build_model(img_size)
print('Model is built')
model.summary()

model.load_weights(model_path)

## COMPILE THE MODEL
print('Compile the model...')

# LossWeights = [1,1]
# lr = tf.keras.callbacks.LearningRateScheduler(custom_lr)
# terminateNaN = TerminateOnNaN()
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
# log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# tqdm_callback = tfa.callbacks.TQDMProgressBar()

# schedule = optimizers.schedules.PolynomialDecay(
#                 initial_learning_rate=learning_rate,
#                 decay_steps=0.0005,
#                 power=0.9
#             )

tf.random.set_seed(40)
# optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
# bin_loss = SparseCategoricalFocalLoss(gamma=2)
# optimizer = tf.keras.optimizers.Adam()
model.compile(loss=[SparseCategoricalFocalLoss(gamma=2), instance_loss],
                  loss_weights=loss_weights,
                  metrics={'bise_net_v2_1': CalculateIOU(2)})

## EVALUATE THE MODEL
print('Evaluating the model...')
model.evaluate(test_dataset)