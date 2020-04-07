# -*- coding: utf-8 -*-
# This is the code for S. Wang, S. Bi and Y. A. Zhang, "Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach," in IEEE Internet of Things Journal.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import losses
#from keras import backend as K
import keras


def test2train(save_name):
    y_pred = sio.loadmat(save_name)['output_mode_pred']
    return y_pred
    # If you want to save the result.
    # sio.savemat(save_name, {'input_h': X_test/10000000,'output_mode':Y_test,'output_mode_pred': y_pred})
        
           
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

def weight_loss(a,b):#Self-defined loss function to handle the unbalance labels
    import tensorflow as tf
    mask_a=tf.greater_equal(a,0.5)
    mask_b=tf.less(a,0.5)
    return (5*losses.binary_crossentropy(tf.boolean_mask(a,mask_a),tf.boolean_mask(b,mask_a))+losses.binary_crossentropy(tf.boolean_mask(a,mask_b),tf.boolean_mask(b,mask_b)))/6


import scipy.io as sio 
# Load data
x_train = sio.loadmat('./data/data118_traintest_big')['x_train']
y_train= sio.loadmat('./data/data118_traintest_big')['y_train']
x_test = sio.loadmat('./data/data118_traintest_big')['x_test']
y_test = sio.loadmat('./data/data118_traintest_big')['y_test']


# Define the network struture
#In this example, the network is 4 layers 1DCNN + 1 Flatten Layer + 1 Fully Connected Layer
model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(x_train.shape[1], 1)))
#model.add(Dropout(0.05))
#model.add(Conv1D(256, 3, activation='relu'))
#model.add(Conv1D(256, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(180, activation='sigmoid'))



# Choose the loss function
# =============================================================================
# model.compile(loss=weight_loss,
#               optimizer='adam',
#               metrics=['accuracy'])
# =============================================================================
model.compile(loss='binary_crossentropy',
               optimizer='adam',
              metrics=['accuracy'])

# Train, evaluate, predict
import numpy as np
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
model.fit(np.expand_dims(x_train,axis=2), y_train, batch_size=100, epochs=200,callbacks=[reduce_lr])
score = model.evaluate(np.expand_dims(x_test,axis=2), y_test, batch_size=100)
pred_y=model.predict(np.expand_dims(x_test,axis=2), batch_size=100)

# The threshold can be changed to generate ROC curve, in this file, the threshold is set as 0.5
for i in range(2000):
    for j in range (180):
        if pred_y[i][j]>0.5:
            pred_y[i][j]=1
        else:
            pred_y[i][j]=0
row,acca=cal_acc(pred_y,y_test)

#Save the result
sio.savemat('./data/118caseresult_weighted', {'output_mode':pred_y,'output_mode_pred': y_test})