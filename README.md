# Study_Pytorch

## Outline
||||||
|---|---|---|---|---|
|[1. Goal](#1-goal)|[2. Using Google colab](#2-using-google-colab)|[3. Install Pytorch - With PyCharm](#3-install-pytorch---with-pycharm)|[4. Study net](#4-study-net)|[5. Author](#5-Author)|


## 1. Goal

| | | |
| :------------ | :-----------: | :-----------: |
|  **Classfication**| | |
|VGG Net|EfficientNet||
| **Object detection** | with COCO Dataset | |
|YOLO|Mask R-CNN||
| **Segmentation** |||
|U-Net|||
| **Generative model** | | |
|GAN|CycleGAN|StarGAN|


- [ ] VGG Net
- [ ] GAN
- [ ] CycleGAN
- [ ] U-Net
- [ ] StarGAN
- [ ] EfficientNet
- [ ] YOLO
- [ ] Mask R-CNN with COCO Dataset

[...Outline](#outline)

## 2. Using Google colab


```
from google.colab import drive
drive.mount('/content/drive/')

!pwd
!ls

cd Pytorch_Study

!git clone https://github.com/nh9k/tutorials.git

cd tutorials/beginner_source/blitz

!python3 cifar10_tutorial.py
```

[...Outline](#outline)

## 3. Install Pytorch - With PyCharm

you can see contents about installing pytorch here!
[myNaverBlog Link](https://blog.naver.com/kimnanhee97/221859176834)

[...Outline](#outline)

## 4. Study net

|model|||||
|---|---|---|---|---|
|[LeNet5](#LeNet5) |||||

[...Outline](#outline)  

## LeNet5

![lenet-5](https://user-images.githubusercontent.com/56310078/77683274-7afad000-6fdb-11ea-8263-9792c3c583d7.png)

[LeNet5 paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
[cifar10 information](https://www.cs.toronto.edu/~kriz/cifar.html)

### (1) Prerequisites

- numpy>=1.18.1
- torch>=1.4.0
- torchvision>=0.5.0
- Pillow==7.0.0

### (2) Usage

**Train a model**
```
python train.py
```
**Test the model**
```
python eval.py
```

### (3) Results

|Cifar10|
|---|
|![result](https://user-images.githubusercontent.com/56310078/77675920-f4d98c00-6fd0-11ea-85c5-40659118e875.JPG)|
|GroundTruth:    cat  ship  ship plane
Predicted:    cat   car  ship plane|

```
Accuracy of the network on the 10000 test images: 54 %
Accuracy of plane : 61 %
Accuracy of   car : 65 %
Accuracy of  bird : 49 %
Accuracy of   cat : 50 %
Accuracy of  deer : 29 %
Accuracy of   dog : 34 %
Accuracy of  frog : 63 %
Accuracy of horse : 62 %
Accuracy of  ship : 58 %
Accuracy of truck : 67 %
```
[...4.Study net](#4-study-net) 

## VGGNet

<img src="https://user-images.githubusercontent.com/56310078/77938303-80fcf380-72f0-11ea-9695-2df938f62df2.JPG" height =500>

[VGGNet paper](https://arxiv.org/pdf/1409.1556.pdf)
[a model pre-trained on ImageNet](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)

Dataset: cifar10
`32x32` -> `16x16` -> `8x8` -> `4x4` -> `2x2` -> `1x1` with 5 maxpooling and same padding(conv2d)

[...4.Study net](#4-study-net)

## GAN

[GAN](https://arxiv.org/pdf/1406.2661.pdf)

## 5. Author
Nanhee Kim / [@nh9k ](https://github.com/nh9k)