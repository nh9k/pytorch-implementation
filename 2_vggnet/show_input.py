from utils import *

if __name__ == '__main__':
    trainloader, _ , _ = build_datasets()

    for epoch in range(2):  # 데이터셋을 수차례 반복합니다.
        for i, data in enumerate(trainloader, 0):
            # 입력을 받은 후,
            inputs, labels = data
            imshow(torchvision.utils.make_grid(inputs))
