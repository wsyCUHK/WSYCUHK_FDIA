# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:54:03 2018

@author: Wang Shuoyao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:18:16 2018

@author: Wang Shuoyao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:36:10 2018

@author: Wang Shuoyao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:28:33 2018

@author: Wang Shuoyao
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:26:39 2018

@author: Wang Shuoyao
"""

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import cnn_kernal_wsy as dnn     # import our function file
import tensorflow as tf


def test2train(save_name):
    y_pred = sio.loadmat(save_name)['output_mode_pred']
    return y_pred
        # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        # # Compute the error for iteration t.
           
def cal_acc(a,b):
    n=a.shape[0]
    m=a.shape[1]
    tterr=0
    r_err=0
    for i in range(n):
        cuerr=0
        for j in range(m):
            if a[i][j]!= b[i][j]:
               tterr+=1
               cuerr+=1
        if cuerr>0:
            r_err+=1
            
    return 1-r_err/n, 1-tterr/(n*m)

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
model_location = "./DNNmodel/model_demo1.ckpt"
save_name="./data/3layer12812864Prediction_%d" % K
#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[128,128,64]
netarray.append(net)
training_epochs=200  #training rate
regularizer=0.0005
batch_size=100
LR = 0.001   #learning rate
in_keep=1
hi_keep=1
LRdecay=1



print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(measurement), training_epochs,len(net)))

# Train the deep neural network
print('train CNN1 ...')
dnn.CNN_train(net,measurement, attack, measurement1, attack1,model_location,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)
dnn.CNN_test(net,measurement1, attack1,  model_location, save_name)



        # return np.argmax(votes, axis=1)


