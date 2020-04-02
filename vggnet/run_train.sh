#!/bin/bash

for model in vgg11 vgg13 vgg16_1 vgg16 vgg19
do
	python3 train.py --arch=$model
done
