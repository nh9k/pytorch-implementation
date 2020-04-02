import argparse
import torch
from torch.autograd import Variable

import vggnet
from utils import *


def eval(trained_model):
    _, testloader, classes = build_datasets()

    ### show some testset result
    dataiter = iter(testloader) #get randomly testsets
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

    outputs = trained_model(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    ### show entire testset result
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = trained_model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    ### what is correct / isn't corret?
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = trained_model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=False, default='vgg11.pth')

    args = parser.parse_args()
    print('started model: ',args.checkpoint_path[:-4])
    trained_lenet = vggnet.__dict__[args.checkpoint_path[:-4]]()
    trained_lenet.load_state_dict(torch.load(args.checkpoint_path))

    eval(trained_lenet)
