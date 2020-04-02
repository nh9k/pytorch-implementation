import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_trainsets():
    trainloader, _ , classes = build_datasets()

    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(trainloader) #https://dojang.io/mod/page/view.php?id=2408
    images, labels = dataiter.next()

    # 정답(label) 출력
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))


def build_datasets():
    '''
    torchvision 데이터셋의 출력(output)은 [0, 1] 범위를 갖는 PILImage 이미지입니다.
    이를 [-1, 1]의 범위로 정규화된 Tensor로 변환하겠습니다.
    '''

    transform = transforms.Compose(
        [transforms.ToTensor(), #image2tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalize using average, root var


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, # 데이터 저장 위치  #True: trainset
                                            download=True, transform=transform) #download 여부  #데이터 선처리 작업
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def Optimizer(Net):
    '''
    define loss func and optimizer
    loss: cross-entropy loss
    optimizer: SGD with momentum
    '''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)

    return criterion, optimizer