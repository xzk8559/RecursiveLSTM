import tensorflow as tf
import os
import numpy as np
import scipy.io
from scipy import signal
import joblib
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def get_args(index):
    class Args:
        def __init__(self, arglist):
            self.dataDir = arglist[0]
            self.trainset = arglist[1]
            self.validset = arglist[2]
            self.dt = arglist[3]
            self.layers = arglist[4]
            self.past_history = arglist[5]
            self.global_sample_step = arglist[6]
            self.local_sample_step = arglist[7]
            self.window_sliding_step = arglist[8]
            self.EPOCHS = 3500
            self.BATCH_SIZE = 256
            self.initial_lr = 0.001
            self.decay = 0.005
            self.initial_epoch = 0
            self.target_size = 1
            
    if index == 0:
        return Args([
            './data/',
            'dataset_t_2.5_mdof10.mat',
            'dataset_v_2.5_mdof10.mat',
            0.01, 6, 100, 1, 4, 2])
        
    elif index == 1:
        return Args([
            './data/',
            'dataset_t_4_mdof10.mat',
            'dataset_v_4_mdof10.mat',
            0.01, 5, 90, 1, 4, 2])
        
    elif index == 2:
        return Args([
            './data/',
            'dataset_t_2.5_mdof6.mat',
            'dataset_v_2.5_mdof6.mat',
            0.01, 7, 50, 1, 4, 2])
        
    elif index == 3:
        return Args([
            './data/',
            'dataset_t_4_mdof6.mat',
            'dataset_v_4_mdof6.mat',
            0.01, 6, 60, 1, 4, 2])
        
    elif index == 4:
        return Args([
            './data/',
            'dataset_t_4_fem.mat',
            'dataset_v_4_fem.mat',
            0.02, 7, 75, 1, 1, 1])
    

def load_data(args, train=1):
    
    if train:
        file = scipy.io.loadmat(args.dataDir + args.trainset)
    else:
        file = scipy.io.loadmat(args.dataDir + args.validset)
        
    lens = file['lens'] # shape (n, 1)
    eq = file['input_tf'] # shape (n, length)
    d = file['target_tf'] # shape (n, length, 3)
    
    lens = lens // args.global_sample_step
    eq = eq[:, ::args.global_sample_step]
    d = d[:, ::args.global_sample_step]
    
    return eq, d, lens

def get_scaler(eq, d):
    
    eq = eq.reshape((-1, 1))
    d = d.reshape((-1, d.shape[2]))
    temp = np.concatenate((eq, d), axis=1)
    
    scaler = MaxAbsScaler()
    scaler.fit(temp)
    data_max = scaler.max_abs_
    joblib.dump(scaler, os.path.join('./save', "scaler.save"))
    
    return scaler, data_max


def get_dataset_xy(eq, d, lens, scaler, args, inverse=1):
    
    x = []
    y = []
    for i in range(len(eq)):
    
        eq0 = eq[i, :lens[i, 0]-1]
        d0 = d[i, :lens[i, 0]-1]
        
        # shape of eq0 (length,)
        # shape of d0 (length, 3)
        eq0 = eq0.reshape((-1, 1))
        dataset = np.concatenate((eq0, d0), axis=1)
        dataset = scaler.transform(dataset)
    
        x_single, y_single = get_dataset_single(dataset, dataset[:, 1:4], 0, None, args)
        x.append(x_single)
        y.append(y_single)
        
    x = np.concatenate(x)
    y = np.concatenate(y)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    if inverse:
        x, y = add_inverse(x, y)
        
    XY = tf.data.Dataset.from_tensor_slices((x, y))
    XY = XY.cache().shuffle(len(x)).batch(args.BATCH_SIZE).repeat()
    
    print ('Single window of past history : {}'.format(x[0].shape))
    print ('Target response to predict : {}'.format(y[0].shape))
    print('Shape of dataset X : {}'.format(x.shape))
    print('Shape of dataset Y : {}'.format(y.shape))
        
    return XY


def get_dataset_single(dataset, target, start_index, end_index, args):
    data = []
    labels = []
    step = args.local_sample_step
    history_size = args.past_history * step
            
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - args.target_size
  
    for i in range(start_index, end_index, args.window_sliding_step):
        
        indices = range(i-history_size, i, step)

        data.append(dataset[indices])
        
        label = target[i:i+args.target_size, :].reshape(args.target_size*target.shape[1])
        labels.append(label)
  
    return np.array(data), np.array(labels)

def add_inverse(a, b):
    a = np.concatenate((a, -a), axis=0)
    b = np.concatenate((b, -b), axis=0)
    return a, b

def lowpass(data, f, fs):
    wn = 2*f/fs
    b, a = signal.butter(8, wn, 'lowpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def scale_valset(eq_val, d_val, scaler):

    eq_val_flat = eq_val.reshape(eq_val.shape[0]*eq_val.shape[1], -1) # (n*l, 1)
    d_val_flat = d_val.reshape(d_val.shape[0]*d_val.shape[1], -1) # (n*l, 3)
    origin = np.concatenate((eq_val_flat, d_val_flat), axis=1) # (n*l, 4)
    origin = scaler.transform(origin)
    origin = origin.reshape(eq_val.shape[0], eq_val.shape[1], -1) # (n, l, 4)
    body = np.concatenate((origin[:,:,0:1], np.zeros(origin[:,:,1:].shape)), axis=2)
    return origin, body

def evaluate(y_true, y_predict):
    
    # common
    max_true = np.max(np.abs(y_true))
    max_predict = np.max(np.abs(y_predict))
    
    re =  np.abs(max_predict/max_true - 1)
    mae = mean_absolute_error(y_true, y_predict)
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    r = np.corrcoef(y_true, y_predict)[1][0]
    r2 = r2_score(y_true, y_predict)
    
    # weighted
    mean_true = np.mean(y_true)
    abs_true = np.abs(y_true)
    w_true = abs_true/np.max(abs_true)
    
    wvar = np.average(np.square(y_true - mean_true), weights = w_true)
    wmse = np.average(np.square(y_true - y_predict), weights = w_true)
    
    wrmse = np.sqrt(wmse)
    rw2 = 1 - wmse/wvar
    
    return np.array([re, mae, rmse, r, r2, wrmse, rw2])