## Deep Learning based Locational Detection architecture
Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach, which is submitted to IEEE Internet of Things Journal.
## Introduction
This is the code and data for our paper "Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach".
![The structure of Deep Learning based Locational Detection architecture.](https://user-images.githubusercontent.com/37823466/68104232-46478880-ff15-11e9-9e39-759c1d568ada.png)

### Network Design
Convolutional networks are designed to process data that come in the form of multiple arrays. Many data modalities are in the form of multiple arrays: 1D for signals and sequences, including vectors of measurements; 2D for images or audio spectrograms; and 3D for video or volumetric
images. The reason for the architecture of convolutional networks is twofold. First, in array data, local groups
of values are often highly correlated, forming distinctive local motifs that are easily detectable. Second, the local
statistics of images and other signals are invariant to location. In other words, if a motif can appear in one part of the
arrays, it could appear anywhere. Hence, the units at different locations share the same weights through convolution
operations and thus the convolutional layers could detect the same pattern in different parts of the array.
In this paper, if an attack can appear in one part of the measurements, it could appear anywhere. Hence, we
use convolutional layers to detect the same pattern in different parts of the measurements. Besides, the local groups
of measurements are often highly correlated and the local statistics of attacks could be invariant to location. For
example, the real and reactive power for each bus should be balanced. Therefore, the adjacent measurements are
injected with high probability to keep the power balance in order to avoid detection. The inconsistency and cooccurrence dependency couple the measurements leaf by leaf. To decouple the inconsistency and dependency leaf by
leaf, we use several layers of convolutional network layers in this paper.

## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* [Keras](http://keras.io//) 2.0
* [NumPy](http://www.numpy.org/) (>1.12.1)

## Training Data
### The training data file contrains 4 matrices：
./data/data118_traintest.mat: There are 4 variables inside our .mat file, i.e., training_x, training_y, testing_x, testing_y.   
        training_x:110000×180  
        training_y:110000×180  
        testing_x:10000×180  
        testing_y:10000×180  

### The training data is generated as follows:  
Step 1: We first generate 110000 sets of base load data by extending the real-world data through artificially generating the loads on each bus with normal distribution.  
Step 2: We randomly select 10000 sets of loads to implement FDIA.  
Step 3: For each attack, we randomly select a set of target state variables to compromise. In particular, the number of target state variables is uniformly distributed between 2-5 and the indices of target state variables are uniformly chosen from all the state variables.  
Step 4: For each set of load and its particular target state variables, we generate stealthy FDIA according to the min-cut algorithm in [Bi & Zhang 2014] ( S. Bi and Y. J. Zhang, “Using covert topological information for defense against malicious attacks on DC state estimation,” IEEE J. on Selected Areas in Comm., vol. 32, no. 7, pp. 1471–1485, July 2014.).  
Step 5: Due to the dynamic measurement noise, we also append random Gaussian noises to both compromised and uncompromised data.  


## About Authors
Shuoyao Wang, yorksywang@tencent.com :Shuoyao Wang received the B.Eng. degree (with first class Hons.) and the Ph.D degree in information engineering from The Chinese University of Hong Kong, Hong Kong, in 2013 and 2018, respectively. Since 2018, he has been with the Department of Risk Management, CDG, Tencent, Shenzhen, China, where he is currently an Senior Researcher. His research interests include nature language processing, adversarial (reinforcement) learning, graph neural networks, optimization theory, dynamic programming, deep learning and reinforcement learning algorithm in both Smart Grid and WeChatPay Risk Management.

Suzhi Bi, bsz@szu.edu.cn

Ying-jun Angela Zhang, yjzhang@ie.cuhk.edu.hk

## Thank You for Reading!!!
