# Study_Pytorch

## Outline
||||||
|---|---|---|---|---|
|[1. Goal](#1-goal)|[2. Using Google colab](#2-using-google-colab)|[3. Install Pytorch - With PyCharm](#3-install-pytorch---with-pycharm)|[4. Study net](#4-study-net)|[5. Author](#5-Author)|


## 1. Goal

| | | |
| :------------ | :-----------: | :-----------: |
|  **Classfication**| | |
|LeNet|VGG Net|EfficientNet|
| **Object detection** |  | |
|YOLO|Faster R-CNN||
| **Segmentation** |||
|U-Net|Mask R-CNN||
| **Generative model** | | |
|GAN|CycleGAN|StarGAN|

- [x] LeNet
- [x] VGG Net
- [ ] GAN
- [ ] U-Net
- [ ] CycleGAN
- [ ] StarGAN
- [ ] Faster R-CNN
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
|[LeNet5](#LeNet5) |[VGGNet](#VGGNet)|[GAN](#GAN)|||

[...Outline](#outline)  

## LeNet5

![lenet-5](https://user-images.githubusercontent.com/56310078/77683274-7afad000-6fdb-11ea-8263-9792c3c583d7.png)

[LeNet5 paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)/
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

 **test**
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

[VGGNet paper](https://arxiv.org/pdf/1409.1556.pdf)/
[torchvision: a model pre-trained on ImageNet](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)

Dataset: cifar10  
`32x32` -> `16x16` -> `8x8` -> `4x4` -> `2x2` -> `1x1` with 5 maxpooling and same padding(conv2d)

### (1) Prerequisites
- numpy
- torch
- torchvision
- tensorboard
- argparse

### (2) Usage

**Train a model**
```
bash -x run_train.sh
```
**Test the model**
```
bash -x run_eval.sh
```
**Use Tensorboard**
```
tensorboard --logdir=result/tensorboard
```
**Use matplotlib for input image**
```
python3 show_input.py
```
**Use matplotlib for log**
```
python3 log_print.py
```

### (3) Results

|tensorboard|
|---|
|![loss](https://user-images.githubusercontent.com/56310078/78297352-5ae28800-756a-11ea-8a63-9333cb385b4f.gif)|
|![accuracy](https://user-images.githubusercontent.com/56310078/78297345-58802e00-756a-11ea-8c85-d81cd10a5225.gif)|

|matplotlib|
|---|
|![Figure_1](https://user-images.githubusercontent.com/56310078/78297930-9c276780-756b-11ea-81d6-a8be3719d916.jpeg)|
|![Figure_2](https://user-images.githubusercontent.com/56310078/78297932-9d589480-756b-11ea-8242-253d586241cf.jpeg)|

**vgg11 test**
```
Accuracy of the network on the 10000 test images: 75 %
Accuracy of plane : 72 %
Accuracy of   car : 85 %
Accuracy of  bird : 60 %
Accuracy of   cat : 47 %
Accuracy of  deer : 77 %
Accuracy of   dog : 63 %
Accuracy of  frog : 80 %
Accuracy of horse : 76 %
Accuracy of  ship : 87 %
Accuracy of truck : 84 %
```

**vgg13 test**
```
Accuracy of the network on the 10000 test images: 76 %
Accuracy of plane : 72 %
Accuracy of   car : 89 %
Accuracy of  bird : 72 %
Accuracy of   cat : 55 %
Accuracy of  deer : 70 %
Accuracy of   dog : 66 %
Accuracy of  frog : 75 %
Accuracy of horse : 80 %
Accuracy of  ship : 90 %
Accuracy of truck : 82 %
```

**vgg16_1 test**
```
Accuracy of the network on the 10000 test images: 76 %
Accuracy of plane : 82 %
Accuracy of   car : 92 %
Accuracy of  bird : 66 %
Accuracy of   cat : 47 %
Accuracy of  deer : 74 %
Accuracy of   dog : 78 %
Accuracy of  frog : 72 %
Accuracy of horse : 88 %
Accuracy of  ship : 87 %
Accuracy of truck : 71 %
```

**vgg16 test**
```
Accuracy of the network on the 10000 test images: 77 %
Accuracy of plane : 79 %
Accuracy of   car : 85 %
Accuracy of  bird : 81 %
Accuracy of   cat : 55 %
Accuracy of  deer : 66 %
Accuracy of   dog : 60 %
Accuracy of  frog : 86 %
Accuracy of horse : 92 %
Accuracy of  ship : 90 %
Accuracy of truck : 69 %
```

**vgg19 test**
```
Accuracy of the network on the 10000 test images: 76 %
Accuracy of plane : 65 %
Accuracy of   car : 82 %
Accuracy of  bird : 60 %
Accuracy of   cat : 58 %
Accuracy of  deer : 66 %
Accuracy of   dog : 60 %
Accuracy of  frog : 83 %
Accuracy of horse : 80 %
Accuracy of  ship : 87 %
Accuracy of truck : 84 %
```

[...4.Study net](#4-study-net)

## GAN

[GAN paper](https://arxiv.org/pdf/1406.2661.pdf)   

[...4.Study net](#4-study-net)

## 5. Author
Nanhee Kim / [@nh9k ](https://github.com/nh9k)