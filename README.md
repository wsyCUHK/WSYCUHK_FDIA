## Deep Learning based Locational Detection architecture
Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach, which is submitted to IEEE Internet of Things Journal.
## Introduction
This is the code and data for our paper "Locational Detection of False Data Injection Attack in Smart Grid: a Multi-label Classification Approach".

## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* [Keras](http://keras.io//) 2.0
* [NumPy](http://www.numpy.org/) (>1.12.1)

## Training Data
# The training data file contrains 4 matrices：
./data/data118_traintest.mat: There are 4 variables inside our .mat file, i.e., training_x, training_y, testing_x, testing_y.   
        training_x:110000×180  
        training_y:110000×180  
        testing_x:10000×180  
        testing_y:10000×180  

# The training data is generated as follows:  
Step 1: We first generate 110000 sets of base load data by extending the real-world data through artificially generating the loads on each bus with normal distribution.  
Step 2: We randomly select 10000 sets of loads to implement FDIA.  
Step 3: For each attack, we randomly select a set of target state variables to compromise. In particular, the number of target state variables is uniformly distributed between 2-5 and the indices of target state variables are uniformly chosen from all the state variables.  
Step 4: For each set of load and its particular target state variables, we generate stealthy FDIA according to the min-cut algorithm in [Bi & Zhang 2014] ( S. Bi and Y. J. Zhang, “Using covert topological information for defense against malicious attacks on DC state estimation,” IEEE J. on Selected Areas in Comm., vol. 32, no. 7, pp. 1471–1485, July 2014.).  
Step 5: Due to the dynamic measurement noise, we also append random Gaussian noises to both compromised and uncompromised data.  


## About Authors
Shuoyao Wang, yorksywang@tencent.com :The first author received the B.Eng. degree (with first class Hons.) and the Ph.D degree in information engineering from The Chinese University of Hong Kong, Hong Kong, in 2013 and 2018, respectively.  His research interests include network embedding, optimization theory, queueing analysis, dynamic programming, deep learning  and reinforcement learning algorithm in Smart Grid.
He is currently a Senior Researcher  on data mining (e.g., NLU, Adversarial Reinforcement Learning, Graph Neural Networks) for Wechat Pay（微信支付） risk management  with Tencent, Shenzhen, China. For any inquiry for this work or cooperation， please feel free to contact us through github.

Suzhi Bi, bsz@szu.edu.cn

Ying-jun Angela Zhang, yjzhang@ie.cuhk.edu.hk

## Thank You for Reading!!!
