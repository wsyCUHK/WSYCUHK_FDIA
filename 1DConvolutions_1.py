# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:23:03 2018

@author: Wang Shuoyao
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

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

import scipy.io as sio 
# Load data
x_train = sio.loadmat('./data/20_train%d' %K)['X_t']
y_train= sio.loadmat('./data/20_train%d' %K)['label_t']
x_test = sio.loadmat('./data/20_train%d' %K)['X_s']
y_test = sio.loadmat('./data/20_train%d' %K)['label_s']


model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.05))
model.add(Conv1D(64, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(19, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

import numpy as np

model.fit(np.expand_dims(x_train,axis=2), y_train, batch_size=100, epochs=100)
score = model.evaluate(np.expand_dims(x_test,axis=2), y_test, batch_size=100)
pred_y=model.predict(np.expand_dims(x_test,axis=2), batch_size=100)
for i in range(4000):
    for j in range (19):
        if pred_y[i][j]>0.5:
            pred_y[i][j]=1
        else:
            pred_y[i][j]=0
print(cal_acc(pred_y,y_test))