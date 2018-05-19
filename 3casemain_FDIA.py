import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_kernal_wangshuoyao as dnn     # import our function file


KK = 1                     # number of users
K= 1

# Load data
measurement = sio.loadmat('./data/3case_embed_train%d' %K)['X_t']
attack = sio.loadmat('./data/3case_embed_train%d' %K)['label_t']
measurement1 = sio.loadmat('./data/3case_embed_train%d' %KK)['X_s']
attack1 = sio.loadmat('./data/3case_embed_train%d' %KK)['label_s']


# Save & Load model from this path
model_location = "./DNNmodel/123model_demo.ckpt"
save_name="./data/123Prediction_%d" % K
#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[120,130]
training_epochs=200  #training rate
regularizer=0.0005
batch_size=100
LR = 0.001   #learning rate
in_keep=1
hi_keep=1
LRdecay=1



print('Case: K=%d, Total Samples: %d, Total Iterations: %d, layers:%d\n'%(K, len(measurement), training_epochs,len(net)))

# Train the deep neural network
print('train DNN ...')
dnn.DNN_train(net,measurement, attack, measurement1, attack1,model_location,export_weight_biase_sw,regularizer,training_epochs,batch_size,LR,in_keep,hi_keep,LRdecay)

# Test Deep Neural Networks
dnntime, Y_pred = dnn.DNN_test(net,measurement1, attack1,  model_location,save_name,binary=1)
print('Testing Time: %0.3f s' % (dnntime))

