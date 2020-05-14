#!/bin/bash

LIST_PTH="
vgg11.pth
vgg13.pth
vgg16_1.pth
vgg16.pth
vgg19.pth
"

for modelpth in $LIST_PTH
do
	python3 eval.py --checkpoint_path=$modelpth
done
