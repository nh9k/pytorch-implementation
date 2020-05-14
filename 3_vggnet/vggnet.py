'''
Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
and https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
'''
import torch.nn as nn
import torch.nn.functional as func
import math

class vggnet(nn.Module):
    '''
    input: 3x32x32 image
    output: 10 class probability
    '''
    def __init__(self, features):
        super(vggnet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(sum(x[0, :]).detach().numpy()[0, 0],
        #       sum(x[1, :]).detach().numpy()[0, 0],
        #       sum(x[2, :]).detach().numpy()[0, 0],
        #       sum(x[3, :]).detach().numpy()[0, 0])
        x = x.view(-1, 512*1*1) #x = x.view(-1, x.size(0))
        x = self.classifier(x)
        # print(sum(x[0, :]).detach().numpy(),
        #       sum(x[1, :]).detach().numpy(),
        #       sum(x[2, :]).detach().numpy(),
        #       sum(x[3, :]).detach().numpy())
        return x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if (v==257) or (v ==513):
                conv2d = nn.Conv2d(in_channels, v-1, kernel_size=1)
                in_channels = v-1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                in_channels = v
            layers += [conv2d, nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 257, 'M', 512, 512, 513, 'M', 512, 512, 513, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def vgg11(): # configuration A
    return vggnet(make_layers(cfgs['A']))

def vgg13(): # configuration B
    return vggnet(make_layers(cfgs['B']))

def vgg16_1(): # configuration C
    return vggnet(make_layers(cfgs['C']))

def vgg16(): # configuration D
    return vggnet(make_layers(cfgs['D']))

def vgg19(): # configuration E
    return vggnet(make_layers(cfgs['E']))