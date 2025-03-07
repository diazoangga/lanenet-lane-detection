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
import matplotlib.pyplot as plt

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
dataset_path = CFG.DATASET.TRAIN_FILE_LIST
num_samples = CFG.DATASET.MAX_NUM_SAMPLES
img_size = CFG.DATASET.IMAGE_SIZE
val_ratio = CFG.DATASET.VAL_RATIO
num_epochs = CFG.TRAIN.EPOCH_NUMS
batch_size = CFG.TRAIN.BATCH_SIZE
val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
learning_rate = CFG.SOLVER.LR
loss_weights = CFG.TRAIN.LOSS_WEIGHTS
save_path = CFG.TRAIN.MODEL_SAVE_DIR
continue_train = CFG.TRAIN.RESTORE_FROM_CHECKPOINT.ENABLE
model_path = CFG.TRAIN.RESTORE_FROM_CHECKPOINT.WEIGHT_PATH
init_epoch = CFG.TRAIN.RESTORE_FROM_CHECKPOINT.START_EPOCH
# warmup_epochs = CFG.TRAIN.WARMUP_EPOCHS


## IMPORT ALL THE DATASETS
print('Importing the datasets with the following parameters...')
print('   Dataset path                    :', dataset_path)
print('   Max number of training data     :', num_samples)
print('   Val to all dataset ratio        :', val_ratio)

img_path, bin_path, inst_path = read_dataset(dataset_path, num_samples=num_samples)
img_train, img_val, bin_train, bin_val, inst_train, inst_val = train_test_split(img_path, bin_path, inst_path, test_size=val_ratio)

train_ds = tf.data.Dataset.from_tensor_slices((img_train, bin_train, inst_train))
val_ds = tf.data.Dataset.from_tensor_slices((img_val, bin_val, inst_val))

print(f'\nImporting the datasets with {len(list(train_ds))} training data and {len(list(val_ds))} validation data is completed')


## CHECK IF THE PREPROCESSING FUNCTION WORKS PROPERLY
train_dataset = train_ds.map(train_preprocessing).cache().shuffle(100).repeat(num_epochs).prefetch(tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True)
val_dataset = val_ds.map(train_preprocessing).cache().repeat(num_epochs*2).prefetch(tf.data.AUTOTUNE).batch(val_batch_size, drop_remainder=True)

train_dataset = iter(train_dataset)
val_dataset = iter(val_dataset)


## BUILD THE MODEL
tf.random.set_seed(40)
model = BiseNetV2().build_model(img_size)
print('Model is built')
model.summary()

if continue_train == True:
    model.load_weights(model_path)
    epoch = init_epoch
else:
    epoch = 0


# TRAIN THE MODEL
print('Setting the training solver...')

date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = ops.join(save_path, date_time)
if not ops.exists(save_path):
    os.makedirs(save_path)

# LossWeights = [1,1]
# lr = tf.keras.callbacks.LearningRateScheduler(custom_lr)
terminateNaN = TerminateOnNaN()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
log_dir = "./logs/fit/" + date_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tqdm_callback = tfa.callbacks.TQDMProgressBar()

schedule = optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=0.0005,
                power=0.9
            )

tf.random.set_seed(40)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
# bin_loss = SparseCategoricalFocalLoss(gamma=2)
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
                  loss=[SparseCategoricalFocalLoss(gamma=2), instance_loss],
                  loss_weights=loss_weights,
                  metrics={'bise_net_v2_1': CalculateIOU(2), 'bise_net_v2_1_1': "accuracy"})

checkpoint = ModelCheckpoint(filepath=ops.join(save_path, 'lanenet_ckpt.epoch{epoch:02d}-loss{loss:.2f}.h5'),
                            monitor='val_loss',
                            verbose=1,
                            save_weights_only=True,
                            save_best_only=True,
                            mode='min')
print('Training is starting')
print(int(np.ceil(len(list(val_ds)) / batch_size)))
history = model.fit(train_dataset,
                   batch_size=batch_size,
                   verbose=1,
                    initial_epoch = epoch,
                   epochs=num_epochs,
                    steps_per_epoch = int(np.floor(len(list(train_ds)) / batch_size)),
                   validation_data=val_dataset,
                    validation_steps= int(np.floor(len(list(val_ds)) / val_batch_size)),
                    shuffle=False,
                   callbacks=[terminateNaN, tensorboard_callback, checkpoint, early_stop])

model.save(ops.join(save_path, 'final.weights.h5'))
np.save(os.path.join(save_path, 'history.npy'), history.history)

def plot_result(save_dir, data1, data2, label1, label2, title):
    x = [i for i in range(len(data1))]

    save_fig_dir = os.path.join(save_dir, 'plot_fig')
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)

    plt.figure(figsize=(12,4))
    plt.plot(x, data1, label=label1)
    plt.plot(x, data2, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, f'{title}.png'))

train_loss_result = history.history['loss']
train_focalloss_result = history.history['bise_net_v2_1_loss']
train_instanceloss_result = history.history['bise_net_v2_1_1_loss']
val_loss_result = history.history['val_loss']
val_focalloss_result = history.history['val_bise_net_v2_1_loss']
val_instanceloss_result = history.history['val_bise_net_v2_1_1_loss']
train_acc_result = history.history['bise_net_v2_1_1_accuracy']
val_acc_result = history.history['val_bise_net_v2_1_1_accuracy']
train_iou_result = history.history['bise_net_v2_1_calculate_iou']
val_iou_result = history.history['val_bise_net_v2_1_calculate_iou']

plot_result(save_path, train_loss_result, val_loss_result, 'Training', 'Validation', 'Training and Validation Loss')
plot_result(save_path, train_focalloss_result, val_focalloss_result, 'Training', 'Validation', 'Training and Validation Focal Loss')
plot_result(save_path, train_instanceloss_result, val_instanceloss_result, 'Training', 'Validation', 'Training and Validation Instance Loss')
plot_result(save_path, train_acc_result, val_acc_result, 'Training', 'Validation', 'Training and Validation Accuracy')
plot_result(save_path, train_iou_result, val_iou_result, 'Training', 'Validation', 'Training and Validation IoU')

# for idx, data in enumerate(val_ds):
#     print(idx)

# for epoch in range(num_epochs):
#     print("\nStart of epoch %d..." %(epoch,))

#     #Iterate over the batches of the dataset.
#     for step in range(3):
#         img_batch_true, bin_batch_true, inst_batch_true = train_dataset.next()
#         print(img_batch_true.shape)
#         with tf.GradientTape() as tape:
#             bin_batch_pred, inst_batch_pred = model(img_batch_true, training=True)
#             total_loss = bin_loss(bin_batch_true, bin_batch_pred) + instance_loss(inst_batch_true, inst_batch_pred)
#         grads = tape.gradient(total_loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         print(f"Total loss for step {step}: {total_loss}")




print(f'\nTraining is completed successifully.\nThe saved data is located in {save_path}')

# SAVE THE MODEL


# Evaluate the model