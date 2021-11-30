import os
import scipy.io
import numpy as np
import tensorflow as tf
print(tf.__version__)

from utils_data import *
from get_model import *


args = get_args(0)
args.dt = 0.005
args.layers = 5
args.past_history = 100 # (75, 125)/global_sample_step
args.global_sample_step = 1
args.local_sample_step = 6
args.window_sliding_step = 1

mat = scipy.io.loadmat('./data/data_MRFDBF.mat')
train_indices = mat['trainInd'] - 1
valid_indices = mat['valInd'] - 1
test_indices = mat['testInd'] - 1
eq_train = mat['input_tf'][train_indices[0], ::args.global_sample_step].astype(np.float32)
eq_valid = mat['input_tf'][valid_indices[0], ::args.global_sample_step].astype(np.float32)
d_train = mat['target_tf'][train_indices[0], ::args.global_sample_step].astype(np.float32)
d_valid = mat['target_tf'][valid_indices[0], ::args.global_sample_step].astype(np.float32)
lens_t = np.array(eq_train.shape[0]*[5001], dtype=np.uint16).reshape((-1,1))
lens_v = np.array(eq_valid.shape[0]*[5001], dtype=np.uint16).reshape((-1,1))

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
                    # initial_epoch = 3000,
                    callbacks=[lrate, checkpoint]) # WandbCallback()

min_vloss = min(history.history['val_loss'])
print('Reached minimum val loss %e at epoch %d.'
      %(min_vloss, history.history['val_loss'].index(min_vloss)))
np.savetxt('./save/loss.txt', history.history['loss'])
np.savetxt('./save/vloss.txt', history.history['val_loss'])

