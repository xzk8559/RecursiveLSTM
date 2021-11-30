import os
import numpy as np
import tensorflow as tf
print(tf.__version__)

from utils_data import *
from get_model import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run LSTM")
    parser.add_argument('--task', type=int, default=0)
    return parser.parse_args()


'''
Load data
eq shape: (num_eq, length_eq)
 d shape: (num_eq, length_eq, 3)
'''
args = get_args(parse_args().task)
eq_train, d_train, lens_t = load_data(args, train=1)
eq_valid, d_valid, lens_v = load_data(args, train=0)
print('Data loaded.')

# Scale
scaler, scaler_max = get_scaler(eq_train, d_train)
print('Data scaled.')

# Generate XY dataset
XY_train = get_dataset_xy(eq_train, d_train, lens_t, scaler, args)
XY_valid = get_dataset_xy(eq_valid, d_valid, lens_v, scaler, args)
print('XY Data generated.')

# Create model
model = get_model(args)

for x, y in XY_valid.take(1):
    print (x.shape)
    print (model(x).shape)
print(model.summary())

initial_lr = args.initial_lr
decay = args.decay
initial_epoch = args.initial_epoch
def step_decay(epoch):
    lr = initial_lr / (1 + decay * (initial_epoch + epoch))
    if lr<5e-5:
        lr = 5e-5
    return lr

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
filepath = os.path.join('./save', "checkpoint.ckpt")
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=0, save_weights_only=True, mode='auto', period=1)
history = model.fit(XY_train,
                    epochs=args.EPOCHS,
                    steps_per_epoch=100,
                    validation_data=XY_valid,
                    validation_steps=50,
                    callbacks=[lrate, checkpoint])

min_vloss = min(history.history['val_loss'])
print('Reached minimum val loss %e at epoch %d.'
      %(min_vloss, history.history['val_loss'].index(min_vloss)))
np.savetxt('../save/loss.txt', history.history['loss'])
np.savetxt('../save/vloss.txt', history.history['val_loss'])

