from UNet import UNet
import argparse
import os
import numpy as np
from CustomDataset import SaveCutTIFData, CustomDataset, ToTensor, Normalization, RandomFlip

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.utils import save_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, help='batch size,  default: 4')
    parser.add_argument('--num_workers', default=4, help='num workers,  default: 2')
    parser.add_argument('--data_dir', default='./datasets', help='data path,  default: ./datasets')
    parser.add_argument('--result_dir', default='./output', help='data path,  default: ./output')
    parser.add_argument('--resume', action='store_true', help='using saved model,  default: True')
    args = parser.parse_args()

    ## select option
    args.resume = True # no saved model
    print(args)

    ## for dataset. if you wanna showing the datasets, input zero in the SaveCutTIFData function
    SaveCutTIFData(1)

    ## for data loader
    transforms = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    test_set = CustomDataset(dir_data=os.path.join(args.data_dir, 'test-cut'), transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ## for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet().to(device)

    ## if resume true
    if args.resume == True:
        net.load_state_dict(torch.load(os.path.join(args.result_dir, 'UNet.pth')))

    ## for Loss function
    fn_loss = nn.BCEWithLogitsLoss().to(device)

    with torch.no_grad():
        net.eval()
        loss_arr = []
        for batch, data in enumerate(test_loader, 0):
            ## forward pass & no backward pass
            labels, inputs = data['label'].to(device), data['input'].to(device)
            outputs = net(inputs)

            ## loss function
            loss = fn_loss(outputs, labels)
            loss_arr += [loss.item()]

            ## print loss
            print("test: [Batch %d/%d] [Loss: %.3f]"
                % (batch, len(test_loader), np.mean(loss_arr)))

            print('shape:', labels.shape, inputs.shape, outputs.shape)
            print('outputs', outputs)
            for i in range(labels.shape[0]):
                save_image(inputs[i].float(), (os.path.join(args.result_dir, "inputs%d.png") % i), normalize=True)
                save_image(labels[i].float(), (os.path.join(args.result_dir,"labels%d.png") % i))
                save_image(outputs[i].float(), (os.path.join(args.result_dir, "outputs%d.png") % i))


    print('Finished Test')

