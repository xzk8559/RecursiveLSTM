import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
print(tf.__version__)

from utils_data import *
from get_model import *

#
args = get_args(parse_args().task)

# Load data
eq_valid, d_valid, lens_v = load_data(args, train=0)
print('Data loaded.')

# Scale
scaler = joblib.load('./save/scaler.save')
print('Scaler loaded.')

# Create model
model = get_model(args)
model.load_weights('./save/checkpoint.ckpt')
print(model.summary())

# Convert Keras to onnx
import keras2onnx
import onnx
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, './save/model-best.onnx')

# Load onnx model
import onnx
import onnxruntime as rt
onnx_model = onnx.load('./save/model-best.onnx')
onnx.checker.check_model(onnx_model)
sess = rt.InferenceSession('./save/model-best.onnx')

#%%
import time
# from utils import lowpass

is_revised = 1
is_filted = 1

inp = args.past_history
step = args.local_sample_step
history_size = inp * step

nval = eq_valid.shape[0]
iteration = eq_valid.shape[1]

origin, body = scale_valset(eq_valid, d_valid, scaler)
        
head = np.zeros((nval, history_size, 4))
result = np.concatenate((head, body), axis=1)

tick1 = time.time()

for i in range(history_size, history_size+iteration):
    
    indices = range(i-history_size, i, step)
    seq = result[:, indices, :]
    
    if is_revised == 0:
        outputs = sess.run(None, {sess.get_inputs()[0].name: seq.astype(np.float32)})[0] # (batch_size, 3)
    elif is_revised == 1:
        seq = np.concatenate((seq, -seq), axis=0)
        outputs = sess.run(None, {sess.get_inputs()[0].name: seq.astype(np.float32)})[0] # (batch_size, 3)
        outputs = (outputs[:nval] - outputs[nval:])/2
    
    result[:, i:i+1, 1:4] = np.reshape(outputs, (-1, 1, 3))
    print ("\r processing: {} / {} iterations ({}%)".format(i-history_size+1, iteration, (i-history_size+1)*100//iteration), end="")

tick2 = time.time()

origin = origin.astype(np.float64)
result = result[:, history_size:, :]
print("\n", tick2 - tick1)

if is_filted == 1:
    for index in range(nval):
        for floor in range(3):
            result[index,:,floor+1] = lowpass(result[index,:,floor+1], 8, 100)


IND = 0
floor = 1 # 1, 2, 3
window = range(500, 1000)
window = range(0, lens_v[IND, 0])

l1 = origin[IND, window, floor]
l2 = result[IND, window, floor]

plt.figure(figsize=(20,12))

line_0 = plt.plot(l1, alpha=0.5, label = 'original disp')[0]
line_0.set_color('red')
line_0.set_linewidth(2.0)
line_4 = plt.plot(l2, alpha=0.5, label = 'predicted disp{}-{}'.format(IND, floor))[0]
line_4.set_color('green')
line_4.set_linewidth(2.0)
plt.legend()
plt.show() 

print(np.corrcoef(l1, l2)[1][0])

#%% evaluate

result_inv = result.copy()
origin_inv = origin.copy()
nval = origin.shape[0]

for i in range(nval):
    result_inv[i, :, :] = scaler.inverse_transform(result_inv[i, :, :]).reshape(1, -1, 4)
    origin_inv[i, :, :] = scaler.inverse_transform(origin_inv[i, :, :]).reshape(1, -1, 4)

# origin = origin/1e3
# result = result/1e3

nfloor = 3 # 1 5 9 / 1 3 6
evaluate_results = np.zeros((nval, 7, nfloor))

for index in range(nval):
    
    # eq = origin[index, :lens_v[index,0], 0]
    for floor in range(nfloor):
        
        y_predict = result_inv[index, :lens_v[index,0],floor+1]
        y_test = origin_inv[index, :lens_v[index,0], floor+1]
        
        evaluate_results[index, :, floor] = evaluate(y_test, y_predict)

evaluate_results_mean0 = np.mean(evaluate_results, axis = 0)
evaluate_results_mean2 = np.mean(evaluate_results, axis = 2)
evaluate_results_mean02 = np.mean(evaluate_results_mean2, axis = 0)
