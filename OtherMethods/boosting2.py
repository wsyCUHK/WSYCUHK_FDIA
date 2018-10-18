# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:27:10 2018

@author: Wang Shuoyao
"""

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_kernal_wangshuoyao as dnn     # import our function file
import tensorflow as tf


def test2train(save_name):
    y_pred = sio.loadmat(save_name)['output_mode_pred']
    return y_pred
        # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        # # Compute the error for iteration t.
           


KK = 1                     # number of users
K= 1

# Load data
measurement = sio.loadmat('./data/20_train%d' %K)['X_t']
attack = sio.loadmat('./data/20_train%d' %K)['label_t']
measurement1 = sio.loadmat('./data/20_train%d' %K)['X_s']
attack1 = sio.loadmat('./data/20_train%d' %K)['label_s']
#measurement1 = sio.loadmat('./data/test%d' %KK)['newStateRe']
#attack1 = sio.loadmat('./data/test%d' %KK)['newlabel']


classifiers = []
netarray=[]



# Save & Load model from this path
model_location2 = "./DNNmodel/model_demo2.ckpt"
save_name2="./data/233Prediction_%d" % K
#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[120,80]
netarray.append(net)
training_epochs=300  #training rate
regularizer=0.0005
batch_size=100
LR = 0.001   #learning rate
in_keep=1
hi_keep=1
LRdecay=1
print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(measurement), training_epochs,len(net)))

# Train the deep neural network
print('train DNN2 ...')
dnn.DNN_train(net,measurement, attack, measurement1, attack1,model_location2,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)
dnn.DNN_test(net,measurement1, attack1,  model_location2, save_name2)


