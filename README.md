# PipeSFL: A Fine-Grained Parallelization Framework for Split Federated Learning on Heterogeneous Clients #  
## Introduction ##
This repository contains the codes of the PipeSFL paper accepted by *IEEE TMC*. PipeSFL is a fine-grained parallelization framework for accelerating Split Federated Learning (SFL) on heterogeneous clients. PipeSFL outperforms the state-of-the-art communication framework [SFL](https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning) and SL.  

## Installation ##
### Prerequisites ###
The following prerequisites shoud be installed for this repository:  
* CUDA >= 10.2  
* PyTorch >= 1.8.1  
### Data Processing ###
You can unzip the Cifar100 in /data folder and run the following scripts in Python3 to prepare the dataset:  
```
python ./scripts/process_cifar100.py  
```
### Quick Start ###
You can run the following scripts:  
```
python ./PipeSFLV1_ResNet50_Cifar100.py  
python ./PipeSFLV2_ResNet50_Cifar100.py
```  
Assume that you have four GPUs on a single node and everything works well, you will see that three clients collaboratively training the ResNet50 model with the Cifar100 dataset using the PipeSFL framework, where the cloud server and each client is one GPU, and the Fed server is one CPU.
