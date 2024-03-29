# Prototypical Network

A re-implementation of [Prototypical Network](https://arxiv.org/abs/1703.05175).

With ConvNet-4 backbone on miniImageNet.


### Results in origin project

1-shot: 49.1% (49.4% in the paper)

5-shot: 66.9% (68.2% in the paper)



  
  
  
  
train__bak.py is the origin file, and train.py is modified from train_bak.py.  
***train.py can get more than 50% 1-shot-5-way acc!!!!***  
  
### Results in this project  

1-shot: 50.17% (49.4% in the paper)（you can get the result less than 100 epoch!）  

5-shot: 68.11% (68.2% in the paper)  


## Environment

* python 3.6
* pytorch 1.2.0

## Instructions

1. Download the images: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

2. Make a folder `materials/images` and put those images into it.

`--gpu` to specify device for program.

### 1-shot Train

`python train.py`

### 1-shot Test

`python test.py` 

### 5-shot Train

`python train.py --shot 5 --train-way 30 --save-path ./save/proto-5`

### 5-shot Test

`python test.py --load ./save/proto-5/max-acc.pth --shot 5`  



### to see the acc log

`python show-logresult.py --load ./save/proto-5/trlog`  

## Acknowledgment  

The following repos are added in this work.

cyvius96/prototypical-network-pytorch:  https://github.com/cyvius96/prototypical-network-pytorch  



