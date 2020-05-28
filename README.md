# Study_Pytorch: model implementation

## Outline
||||||
|---|---|---|---|---|
|[1. Goal](#1-goal)|[2. Using Google colab](#2-using-google-colab)|[3. Install Pytorch - With PyCharm](#3-install-pytorch---with-pycharm)|[4. Study nets](#4-study-nets)|[5. Author](#5-Author)|



## 1. Goal

| | | ||
| :------------ | :-----------: | :-----------: |:-----------: |
|  **Classfication & Backbone**||||
|[LeNet5](#LeNet5)|[VGGNet](#VGGNet)|(ResNet)|EfficientNet|
| **Object detection**||||
|YOLO(V3)|Faster R-CNN|||
| **Segmentation**||||
|[U-Net](#U-Net)|(DeepLabV3)|Mask R-CNN||
| **Generative model** ||||
|[GAN](#GAN)|(DCGAN)|CycleGAN|StarGAN(V1/V2)|

- [x] LeNet(2020.04)
- [x] VGG Net(2020.04)
- [ ] (ResNet)
- [x] GAN(2020.05)
- [ ] (DCGAN)
- [ ] U-Net
- [ ] CycleGAN
- [ ] StarGAN(V1/V2)
- [ ] Faster R-CNN
- [ ] EfficientNet
- [ ] (DeepLabV3)
- [ ] YOLO(V3)
- [ ] Mask R-CNN with COCO Dataset

[Go Outline](#outline)


  
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

[Go Outline](#outline)


  
## 3. Install Pytorch - With PyCharm

you can see contents about installing pytorch here!
[myNaverBlog Link](https://blog.naver.com/kimnanhee97/221859176834)

[Go Outline](#outline)


  
## 4. Study nets

|model|||||
|---|---|---|---|---|
|[LeNet5](#LeNet5) |[VGGNet](#VGGNet)|[GAN](#GAN)|||

[Go Outline](#outline)  


  
## LeNet5

![lenet-5](https://user-images.githubusercontent.com/56310078/77683274-7afad000-6fdb-11ea-8263-9792c3c583d7.png)

[[paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
[[cifar10 information]](https://www.cs.toronto.edu/~kriz/cifar.html)
[[model-code]](https://github.com/nh9k/pytorch-implementation/blob/master/1_lenet5/lenet5.py)
[[training-code]](https://github.com/nh9k/pytorch-implementation/blob/master/1_lenet5/train.py)
  
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
[Go 4.Study nets](#4-study-nets) 


  
## VGGNet

<img src="https://user-images.githubusercontent.com/56310078/77938303-80fcf380-72f0-11ea-9695-2df938f62df2.JPG" height =500>

[[VGGNet paper]](https://arxiv.org/pdf/1409.1556.pdf)
[[torchvision: a model pre-trained on ImageNet]](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
[[model-code]](https://github.com/nh9k/pytorch-implementation/blob/master/2_vggnet/vggnet.py) 
[[training-code]](https://github.com/nh9k/pytorch-implementation/blob/master/2_vggnet/train.py)

Dataset: cifar10  
`32x32` -> `16x16` -> `8x8` -> `4x4` -> `2x2` -> `1x1` with 5 maxpooling and same padding(conv2d)


### (1) Prerequisites
- numpy
- torch
- torchvision
- tensorboard
- argparse

### (2) Usage

**Training**
```
bash -x run_train.sh
```
**Test**
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
|![Loss](https://user-images.githubusercontent.com/56310078/81693679-36e46180-949b-11ea-9535-be43ed945515.gif)|
|![Accuracy](https://user-images.githubusercontent.com/56310078/81693675-35b33480-949b-11ea-9ba5-73c2b0cc6cfb.gif)|

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

[Go 4.Study nets](#4-study-nets)


  
## GAN

![GAN_Loss](https://user-images.githubusercontent.com/56310078/81842190-8fdcf400-9586-11ea-9fff-548a54017abd.JPG)
![GAN_Data_training](https://user-images.githubusercontent.com/56310078/81842193-90758a80-9586-11ea-8655-f09d49ee915c.JPG)

[[GAN paper]](https://arxiv.org/pdf/1406.2661.pdf) 
[[model-code]](https://github.com/nh9k/pytorch-implementation/blob/master/3_GAN/versionLeakyReLU/GAN.py) 
[[training-code]](https://github.com/nh9k/pytorch-implementation/blob/master/3_GAN/versionLeakyReLU/train.py)
  
  
Dataset: MNIST 

### (1) Prerequisites

- numpy
- torch
- torchvision
- tensorboard
- argparse

### (2) Usage

**Training**
```
cd GAN
python train.py
```

### (3) Experiments and Issues

1. `BatchNorm` function applied in model `Generator`
2. `Discriminator` layers `simplication`
3. `Model input/output` should be same `image shape`(scale)
4. for training, do care using `fake_imgs.detach()` or `backward(retain_graph=True)`

The above issues are very important for training GAN 

### (4) Results

Base model: using `ReLU`  
vest model: using `LeakyReLU`
  
1. Using `LeakyReLU` instead of `ReLU`

|500|1000|1500|2000|2500|
|---|---|---|---|---|
|![500](https://user-images.githubusercontent.com/56310078/81840814-7470e980-9584-11ea-988b-77f8fcb74d06.png)|![1000](https://user-images.githubusercontent.com/56310078/81840815-75098000-9584-11ea-9df0-fbf812b5b56c.png)|![1500](https://user-images.githubusercontent.com/56310078/81840816-75a21680-9584-11ea-98fa-e15c6ea515a3.png)|![2000](https://user-images.githubusercontent.com/56310078/81840819-763aad00-9584-11ea-956a-dcd1f47b8c3b.png)|![2500](https://user-images.githubusercontent.com/56310078/81840822-763aad00-9584-11ea-88e4-627a8c9466eb.png)|
|40000|60000|80000|90000|Result(gif)|
|![40000](https://user-images.githubusercontent.com/56310078/81840891-91a5b800-9584-11ea-95b4-302057efaa82.png)|![60000](https://user-images.githubusercontent.com/56310078/81840892-91a5b800-9584-11ea-8893-68e61fef46e2.png)|![80000](https://user-images.githubusercontent.com/56310078/81840895-923e4e80-9584-11ea-9135-8c87cda7d287.png)|![90000](https://user-images.githubusercontent.com/56310078/81840886-90748b00-9584-11ea-9120-c127d19587e8.png)|![gifimage](https://user-images.githubusercontent.com/56310078/81841457-6f606a00-9585-11ea-9771-906fddb2b78f.gif)|

2. Using `ReLU` instead of `LeakyReLU`

|500|1000|1500|2000|2500|
|---|---|---|---|---|
|![500](https://user-images.githubusercontent.com/56310078/81838175-bef06700-9580-11ea-8752-2ad297e7a1a9.png)|![1000](https://user-images.githubusercontent.com/56310078/81838177-bf88fd80-9580-11ea-87d3-c212b5e85ad7.png)|![1500](https://user-images.githubusercontent.com/56310078/81838181-c0219400-9580-11ea-9d3a-ac0bf7cc8b3a.png)|![2000](https://user-images.githubusercontent.com/56310078/81838184-c0219400-9580-11ea-905a-aeb67ea1e0c6.png)|![2500](https://user-images.githubusercontent.com/56310078/81838185-c0ba2a80-9580-11ea-85dd-1b5387123eed.png)|
|20000|40000|60000|80000|Result(gif)|
|![20000](https://user-images.githubusercontent.com/56310078/81838186-c152c100-9580-11ea-9499-7bc43e305fea.png)|![40000](https://user-images.githubusercontent.com/56310078/81838187-c152c100-9580-11ea-8a89-d577948ba6d4.png)|![60000](https://user-images.githubusercontent.com/56310078/81838172-be57d080-9580-11ea-9060-3a07f929dc32.png)|![80000](https://user-images.githubusercontent.com/56310078/81838719-7a190000-9581-11ea-8748-633c20a6f36c.png)|![gifimage](https://user-images.githubusercontent.com/56310078/81839590-bac54900-9582-11ea-88fb-a5500c355f01.gif)|

3. `No BatchNorm` in model

|500|1000|1500|2000|2500|
|---|---|---|---|---|
|![500](https://user-images.githubusercontent.com/56310078/81836203-1d681600-957e-11ea-8c46-86fc0a72bd9a.png)|![1000](https://user-images.githubusercontent.com/56310078/81836208-1e994300-957e-11ea-8fd3-9a858dc6cc24.png)|![1500](https://user-images.githubusercontent.com/56310078/81836209-1e994300-957e-11ea-8097-83f5dec72f9a.png)|![2000](https://user-images.githubusercontent.com/56310078/81836211-1f31d980-957e-11ea-967a-f2140ddaff86.png)|![2500](https://user-images.githubusercontent.com/56310078/81836212-1f31d980-957e-11ea-95fc-06134a3a01f7.png)|

4. `No Simplication` of Discriminator layers

|500|1000|1500|2000|2500|
|---|---|---|---|---|
|![500](https://user-images.githubusercontent.com/56310078/81836311-425c8900-957e-11ea-9c79-953b98aa95f9.png)|![1000](https://user-images.githubusercontent.com/56310078/81836314-438db600-957e-11ea-9f2c-ad7c004ea6f7.png)|![1500](https://user-images.githubusercontent.com/56310078/81836315-438db600-957e-11ea-8426-9e6a4ac3c1bf.png)|![2000](https://user-images.githubusercontent.com/56310078/81836318-44264c80-957e-11ea-9c83-e7259cbdded2.png)|![2500](https://user-images.githubusercontent.com/56310078/81836320-44264c80-957e-11ea-82a4-f18bae9a1452.png)|

5. `backward(retain_graph=True)` instead of `fake_imgs.detach()`

|500|1000|1500|2000|2500|
|---|---|---|---|---|
|![500](https://user-images.githubusercontent.com/56310078/81836880-083fb700-957f-11ea-96bc-8a929590ee0a.png)|![1000](https://user-images.githubusercontent.com/56310078/81836883-083fb700-957f-11ea-9166-445330da8220.png)|![1500](https://user-images.githubusercontent.com/56310078/81836886-08d84d80-957f-11ea-82d5-711e9f2c1182.png)|![2000](https://user-images.githubusercontent.com/56310078/81836888-0970e400-957f-11ea-8941-320477e14e24.png)|![2500](https://user-images.githubusercontent.com/56310078/81836875-07a72080-957f-11ea-984e-6782909bf7c6.png)|

  
[Go 4.Study net](#4-study-nets)

## U-Net

![Unet](https://user-images.githubusercontent.com/56310078/81949340-b9a52200-963d-11ea-8b32-d247d4b5c042.JPG)

[[U-Net Paper]](https://arxiv.org/pdf/1505.04597.pdf) 
[[The full implementation (based on Caffe) and the trained networks]](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)  
  
Dataset: [isbi_challenge](http://brainiac2.mit.edu/isbi_challenge/home)  
you can download segmentation dataset after join in the page.  

(1) Usage  

* Training : [code](#)
```
cd U-Net
python train.py
```

* Test : [code](#)
```
cd U-Net
python eval.py
```

* Tensorboard : [code](#)
```
cd U-Net
tensorboard --logdir=runs
```
![Loss_tensorboard](https://user-images.githubusercontent.com/56310078/83134531-778cdd80-a11f-11ea-9268-58246a7c3c2e.PNG)  

(2) Experiments and Results   
`Cross Entropy Loss` is correct because the output segmentation map channels of the last layer is `2` in the paper, but i modified it to `1` and used `Binary Cross Entropy`  

Test: [Loss: 0.239]  

|Inputs|Labels|Outputs|
|---|---|---|
|![inputs0](https://user-images.githubusercontent.com/56310078/83134546-7c519180-a11f-11ea-81db-2cb18578eb2d.png)|![labels0](https://user-images.githubusercontent.com/56310078/83134559-7fe51880-a11f-11ea-89a9-0b18daf6e942.png)|![outputs0](https://user-images.githubusercontent.com/56310078/83134554-7eb3eb80-a11f-11ea-9cf5-13e631a9260b.png)|
|![inputs1](https://user-images.githubusercontent.com/56310078/83134548-7d82be80-a11f-11ea-93d7-017d3c9ec879.png)|![labels1](https://user-images.githubusercontent.com/56310078/83134560-7fe51880-a11f-11ea-9b16-c3ff864d49cd.png)|![outputs1](https://user-images.githubusercontent.com/56310078/83134555-7f4c8200-a11f-11ea-9bab-68c963fa61a9.png)|
|![inputs2](https://user-images.githubusercontent.com/56310078/83134551-7e1b5500-a11f-11ea-90b2-71f89142c536.png)|![labels2](https://user-images.githubusercontent.com/56310078/83134561-807daf00-a11f-11ea-9dcb-f5e0ff249608.png)|![outputs2](https://user-images.githubusercontent.com/56310078/83134557-7f4c8200-a11f-11ea-819c-ace3d57a93bc.png)|


[Go 4.Study net](#4-study-nets)
  
## 5. Author
Nanhee Kim / [@nh9k ](https://github.com/nh9k)