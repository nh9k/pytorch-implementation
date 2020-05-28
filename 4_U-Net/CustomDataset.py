## reference: https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet
from PIL import Image #pillow
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

## for saving cut-tif-data: http://brainiac2.mit.edu/isbi_challenge/home
def SaveCutTIFData(Show_dataset_False):
    dir_data = './datasets'

    ## image read
    img_label = Image.open(os.path.join(dir_data, 'train-labels.tif'))
    img_input = Image.open(os.path.join(dir_data, 'train-volume.tif'))

    ## image size, frame number
    height, width = img_label.size
    nframe = img_label.n_frames
    nframe_train, nframe_val, nframe_test = int(0.8 * nframe), int(0.1 * nframe), int(0.1 * nframe)

    ## save folder name
    dir_save_train = os.path.join(dir_data, 'train-cut')
    dir_save_val = os.path.join(dir_data, 'val-cut')
    dir_save_test = os.path.join(dir_data, 'test-cut')

    ## create directory
    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)
        os.makedirs(dir_save_val)
        os.makedirs(dir_save_test)

    ## get random sequence frame number
    frame_id = np.arange(nframe)
    np.random.shuffle(frame_id)

    ## save train set
    for i in range(nframe_train):
        img_label.seek(frame_id[i])
        img_input.seek(frame_id[i])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

    ## save validation set
    offset_nframe = nframe_train
    for i in range(nframe_val):
        img_label.seek(frame_id[i + offset_nframe])
        img_input.seek(frame_id[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

    ## save test set
    offset_nframe += nframe_val
    for i in range(nframe_test):
        img_label.seek(frame_id[i + offset_nframe])
        img_input.seek(frame_id[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

    def ShowDataset():
        plt.subplot(121)
        plt.imshow(label_, cmap='gray')
        plt.title('label')

        plt.subplot(122)
        plt.imshow(input_, cmap='gray')
        plt.title('input')

        plt.show()

    if Show_dataset_False == 0:
        ShowDataset()


## Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dir_data, transform=None):
        self.dir_data = dir_data
        self.transform = transform

        list_data = os.listdir(self.dir_data)

        list_label = [f for f in list_data if f.startswith('label')]
        list_input = [f for f in list_data if f.startswith('input')]

        list_label.sort()
        list_input.sort()

        self.list_label = list_label
        self.list_input = list_input

    def __len__(self):
        return len(self.list_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.dir_data, self.list_label[index]))
        input = np.load(os.path.join(self.dir_data, self.list_input[index]))

        ## saved image transform (0~255 range and unsigned 8) to (0~1)
        label = label/255.0
        input = input/255.0

        ## make (x axis, y axis,) channel axis
        if label.ndim == 2:
            label = label[:,:,np.newaxis]
            input = input[:,:,np.newaxis]

        ## return dictionary data type
        data = {'input':input, 'label':label}

        if self.transform:
            return self.transform(data)

        return data

## test Datasets function
def ShowDataset(custom_dataset):
    data = custom_dataset.__getitem__(0)
    input = data['input']
    label = data['label']

    plt.subplot(121)
    plt.imshow(input.squeeze())
    plt.title('input')

    plt.subplot(122)
    plt.imshow(label.squeeze()) #using squeeze, remove dimension that 1
    plt.title('label')

    plt.show()

## Custom Transform
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data