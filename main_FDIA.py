import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import dnn_wsy as dnn     # import our function file


KK = 1                     # number of users
K= 2

# Load data
measurement = sio.loadmat('./data/train%d' %K)['newStateRe']
attack = sio.loadmat('./data/train%d' %K)['newlabel']
measurement1 = sio.loadmat('./data/test%d' %KK)['newStateRe']
attack1 = sio.loadmat('./data/test%d' %KK)['newlabel']
# harvesting_time = sio.loadmat('./data/data_%d' %K)['output_a']
# offloading_time = sio.loadmat('./data/data_%d' %K)['output_tau']
# gain = sio.loadmat('./data/data_%d' %K)['output_obj']

# pre-process data
# offloading_time = mode
# channel = channel * 10000000 # the wireless channel gain is too small, which is scaled up for better training performance.

# split the data samples, train:validation:test = 60:20:20
# split_idx = [int(.6*len(channel)), int(.8*len(channel))]
# X_train, X_valid, X_test = np.split(channel, split_idx)
# mode_train, mode_valid, mode_test = np.split(mode, split_idx)
# Y_train, Y_valid, Y_test = np.split(offloading_time, split_idx)
# gain_train, gain_valid, gain_test = np.split(gain, split_idx)

# Save & Load model from this path
model_location = "./DNNmodel/model_demo.ckpt"
save_name="./data/Prediction_%d" % K

#export the weights and biases or not
export_weight_biase_sw=1

#hyper-parameter
net=[120,130]
training_epochs=200  #what?
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

