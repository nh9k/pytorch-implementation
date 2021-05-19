# Study_Pytorch: model implementation

## Outline
||||||
|---|---|---|---|---|
|[1. Goal](#1-goal)|[2. Using Google colab](#2-using-google-colab)|[3. Install Pytorch - With PyCharm](#3-install-pytorch---with-pycharm)|[4. Study Models](#4-study-models)|[5. Author](#5-Author)|

<br/>  <br/>  


## 1. Goal  

() model: I plan to study, but there is no plan to implement yet

| | | | |
| :------------ | :-----------: | :-----------: |:-----------: |
|  **Classfication & Backbone**||||
|[LeNet5](#LeNet5)|[VGGNet](#VGGNet)|[(ResNet)](#ResNet)|EfficientNet|
| **Object detection**||||
|[Faster R-CNN](#)|[(FPN)](#FPN)|YOLO(V4)||
| **Segmentation**||||
|[(FCN)](#FCN)|[U-Net](#U-Net)|(DeepLab V3, V3+)|[Mask R-CNN](#mask-r-cnn)|
| **Generative model** ||||
|[GAN](#GAN)|(DCGAN)|[CycleGAN](#CycleGAN)|StarGAN(V1,V2)|
  
  
  
Finished  
- [x] LeNet(2020.04)
- [x] VGG Net(2020.04)
- [x] (ResNet)(2020.06)
- [x] GAN(2020.05)
- [ ] (DCGAN)
- [x] (FCN)(2020.07)
- [x] U-Net(2020.05)
- [x] CycleGAN(2020.05)
- [ ] (DeepLabV3)
- [ ] StarGAN(V1/V2)
- [x] Faster R-CNN(2021.05)
- [x] (FPN)(2020.04)
- [ ] EfficientNet
- [x] Mask R-CNN(2020.07)
- [ ] YOLO(V3)


[Go Outline](#outline)  

<br/>  <br/>  


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

<br/>  <br/>  


## 3. Install Pytorch - With PyCharm

you can see contents about installing pytorch here!
[Blog Link](https://blog.naver.com/kimnanhee97/221859176834)

[Go Outline](#outline)  

<br/>  <br/>  


## 4. Study Models

|model-Implementation or using||||||
|---|---|---|---|---|---|
|[LeNet5](#LeNet5) |[VGGNet](#VGGNet)|[GAN](#GAN)|[U-Net](#U-Net)|[Faster R-CNN & Mask R-CNN](#faster-r-cnn&mask-r-cnn)|

|paper-review or study|||||
|---|---|---|---|---|
|[ResNet](#ResNet)|[CycleGAN](#CycleGAN)|[FCN](#FCN)|[FPN](#FPN)||

[Go Outline](#outline)  

<br/>  <br/>  


## LeNet5

![lenet-5](https://user-images.githubusercontent.com/56310078/77683274-7afad000-6fdb-11ea-8263-9792c3c583d7.png)

[[LeNet paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
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
[Go 4.Study Models](#4-study-models)  

<br/>  <br/>  


## VGGNet

<img src="https://user-images.githubusercontent.com/56310078/77938303-80fcf380-72f0-11ea-9695-2df938f62df2.JPG" height =500>

[[VGGNet paper]](https://arxiv.org/pdf/1409.1556.pdf)
[[torchvision: a model pre-trained on ImageNet]](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
[[model-code]](https://github.com/nh9k/pytorch-implementation/blob/master/2_vggnet/vggnet.py) 
[[training-code]](https://github.com/nh9k/pytorch-implementation/blob/master/2_vggnet/train.py)
  
[[Blog - VGGNet, Shell script]](https://blog.naver.com/kimnanhee97/221884194285)

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

[Go 4.Study Models](#4-study-models)  

<br/>  <br/>  


## ResNet
[[ResNet paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
[[model-code of torchvision]](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  

model is not implemented from my code, but I reviewed paper for my study.  
[[Blog - ResNet Summary(Korean)]](https://blog.naver.com/kimnanhee97/222007393892)    

[Go 4.Study Models](#4-study-models)  

<br/>  <br/>  


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
Best model: using `LeakyReLU`
  
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

  
[Go 4.Study Models](#4-study-models)  

<br/>  <br/>  


## FCN
[[FCN paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
[[model-code of torchvision]](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/fcn.py)
[[other model code]](https://github.com/wkentaro/pytorch-fcn)  

model is not implemented from my code, but I studyed roughly.  
[[Blog - FCN Summary(Korean)]](https://blog.naver.com/kimnanhee97/222027492751)      

[Go 4.Study Models](#4-study-models)  

<br/>  <br/>  


## U-Net

![Unet](https://user-images.githubusercontent.com/56310078/81949340-b9a52200-963d-11ea-8b32-d247d4b5c042.JPG)

[[U-Net Paper]](https://arxiv.org/pdf/1505.04597.pdf) 
[[The full implementation (based on Caffe) and the trained networks]](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
[[model-code]](https://github.com/nh9k/pytorch-implementation/blob/master/4_U-Net/UNet.py)
  
[[Blog - UNet, Binary Cross Entropy(Korean)]](https://blog.naver.com/kimnanhee97/221982086442)
[[Blog - UNet, Metrics(Korean)]](https://blog.naver.com/kimnanhee97/221978626236)  
  
Dataset: [isbi_challenge](http://brainiac2.mit.edu/isbi_challenge/home)  
you can download segmentation dataset after join in the page.  
  
### (1) Prerequisites

- numpy
- torch
- torchvision
- Pillow
- matplotlib

### (2) Usage  

* Training : [code](https://github.com/nh9k/pytorch-implementation/blob/master/4_U-Net/train.py)
```
cd U-Net
python train.py
```

* Test : [code](https://github.com/nh9k/pytorch-implementation/blob/master/4_U-Net/eval.py)
```
cd U-Net
python eval.py
```

* Tensorboard
```
cd U-Net
tensorboard --logdir=runs/no_batch_normalization or add_batch_normalization
```

* Jupyter Notebook & Colab :   
[no batch normalization result & code](https://github.com/nh9k/pytorch-implementation/blob/master/4_U-Net/ipynb/no_batch_normalization/unet.ipynb)  
[add batch normalization result & code](https://github.com/nh9k/pytorch-implementation/blob/master/4_U-Net/ipynb/add_batch_normalization/unet.ipynb)

### (3) Experiments and Results   
`Cross Entropy Loss` is correct because the output segmentation map channels of the last layer is `2` in the paper, but i modified it to `1` and used `Binary Cross Entropy`  
  
1. `No` batch normalization
![Loss_tensorboard](https://user-images.githubusercontent.com/56310078/83134531-778cdd80-a11f-11ea-9268-58246a7c3c2e.PNG)  

|Inputs|Labels|Outputs(Test Loss: 0.239)|
|---|---|---|
|![inputs0](https://user-images.githubusercontent.com/56310078/83134546-7c519180-a11f-11ea-81db-2cb18578eb2d.png)|![labels0](https://user-images.githubusercontent.com/56310078/83134559-7fe51880-a11f-11ea-89a9-0b18daf6e942.png)|![outputs0](https://user-images.githubusercontent.com/56310078/83134554-7eb3eb80-a11f-11ea-9cf5-13e631a9260b.png)|
|![inputs1](https://user-images.githubusercontent.com/56310078/83134548-7d82be80-a11f-11ea-93d7-017d3c9ec879.png)|![labels1](https://user-images.githubusercontent.com/56310078/83134560-7fe51880-a11f-11ea-9b16-c3ff864d49cd.png)|![outputs1](https://user-images.githubusercontent.com/56310078/83134555-7f4c8200-a11f-11ea-9bab-68c963fa61a9.png)|
|![inputs2](https://user-images.githubusercontent.com/56310078/83134551-7e1b5500-a11f-11ea-90b2-71f89142c536.png)|![labels2](https://user-images.githubusercontent.com/56310078/83134561-807daf00-a11f-11ea-9dcb-f5e0ff249608.png)|![outputs2](https://user-images.githubusercontent.com/56310078/83134557-7f4c8200-a11f-11ea-819c-ace3d57a93bc.png)|

2. `added` batch normalization
```
## model file: UNet.py (line8~12), added batch normalization function
        def DoubleConv(in_channels, out_channels):
            layers = []
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)
```
![Loss_tensorboard](https://user-images.githubusercontent.com/56310078/83158846-453fa800-a140-11ea-8181-35a1beb6bb56.PNG)  

|Inputs|Labels|Outputs(Test Loss: 0.103)|
|---|---|---|
|![inputs0](https://user-images.githubusercontent.com/56310078/83158396-b3d03600-a13f-11ea-97ae-b0705ad07a5a.png)|![labels0](https://user-images.githubusercontent.com/56310078/83158407-b6329000-a13f-11ea-9d34-579c81d392af.png)|![outputs0](https://user-images.githubusercontent.com/56310078/83158413-b763bd00-a13f-11ea-9cc0-7261e697d624.png)|
|![inputs1](https://user-images.githubusercontent.com/56310078/83158404-b5016300-a13f-11ea-805a-3db8aedd8f1f.png)|![labels1](https://user-images.githubusercontent.com/56310078/83158409-b6cb2680-a13f-11ea-963a-f5ac04d99f31.png)|![outputs1](https://user-images.githubusercontent.com/56310078/83158417-b7fc5380-a13f-11ea-8484-223b3f57a59e.png)|
|![inputs2](https://user-images.githubusercontent.com/56310078/83158406-b599f980-a13f-11ea-8f40-11f34c3398b8.png)|![labels2](https://user-images.githubusercontent.com/56310078/83158410-b6cb2680-a13f-11ea-8b7a-b7d1ab5e48d8.png)|![outputs2](https://user-images.githubusercontent.com/56310078/83158419-b7fc5380-a13f-11ea-8f89-a3dae83a38ad.png)|

[Go 4.Study Models](#4-study-models)  
  
<br/>  <br/>  


## CycleGAN

[[CycleGAN paper]](https://arxiv.org/pdf/1703.10593.pdf)

model is not implemented from my code, but I reviewed paper for my study.  
[[Blog - Replay Buffer(Korean)]](https://blog.naver.com/kimnanhee97/221967906769)
[[Blog - Idea(Korean)]](https://blog.naver.com/kimnanhee97/221988558717)   


[Go 4.Study Models](#4-study-models)   

<br/>  <br/>  


## FPN
[[FPN paper]](https://arxiv.org/pdf/1612.03144.pdf)  

model is not implemented yet from my code, but I studyed roughly.  
[[Blog - FPN research(Korean)]](https://blog.naver.com/kimnanhee97/221933182912)    


[Go 4.Study Models](#4-study-models)   

<br/>  <br/>  


## Faster R-CNN & Mask R-CNN
[[Faster R-CNN paper]](https://arxiv.org/pdf/1506.01497.pdf)
[[Mask R-CNN paper]](https://arxiv.org/pdf/1703.06870.pdf)
[[model-code of Detectron]](https://github.com/facebookresearch/Detectron)
[[Detectron2 github]](https://github.com/facebookresearch/detectron2)   

model is not implemented yet from my code, but I studied detectron2.  
[[Blog - Detection and Segmentation Summary(Korean)]](https://blog.naver.com/kimnanhee97/222026032725)  
[[Webtoon Segmentation using Detectron2(model: Mask R-CNN)]](https://github.com/overfitting-ai-community/basic-course)  

If you want to train model: Faster R-CNN, Just change the model

from
```
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
```

to
```
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
```

if you don't know how to label the dataset,
consider the labeling tool `labelme`

[[labelme github]](https://github.com/wkentaro/labelme)  

  
[Go 4.Study Models](#4-study-models)   

<br/>  <br/>  


## 5. Author
Nanhee Kim [@nh9k](https://github.com/nh9k) / [Personal blog](https://blog.naver.com/kimnanhee97)  