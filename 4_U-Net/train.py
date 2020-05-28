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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, help='batch size,  default: 4')
    parser.add_argument('--num_workers', default=4, help='num workers,  default: 2')
    parser.add_argument('--learning_rate', default=0.0001, help='learning rate,  default: 0.0002')
    parser.add_argument('--b1', default=0.5, help='adam: decay of first order momentum of gradient,  default: 0.5')
    parser.add_argument('--b2', default=0.99, help='adam: decay of second order momentum of gradient,  default: 0.9')
    parser.add_argument('--epochs', default=100, help='epoch,  default: 100')
    parser.add_argument('--data_dir', default='./datasets', help='data path,  default: ./datasets')
    parser.add_argument('--resume', action='store_true', help='using saved model,  default: No')
    args = parser.parse_args()

    ## select option
    args.resume = False # no saved model
    print(args)

    ## for dataset. if you wanna showing the datasets, input zero in the SaveCutTIFData function
    SaveCutTIFData(1)

    ## for data loader
    transforms = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    train_set = CustomDataset(dir_data=os.path.join(args.data_dir, 'train-cut'), transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_set = CustomDataset(dir_data=os.path.join(args.data_dir, 'val-cut'), transform=transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ## for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet().to(device)

    ## if resume true
    if args.resume == True:
        net.load_state_dict(torch.load('/output/UNet.pth'))

    ## save model here
    if not os.path.exists('output'):
        os.makedirs('output')

    ## for Tensorboard
    print(net)
    writer = SummaryWriter(comment='UNet')

    ## for Loss function
    fn_loss = nn.BCEWithLogitsLoss().to(device)

    ## for Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

    ## training
    for epoch in range(args.epochs):
        net.train()
        loss_arr = []
        for batch, data in enumerate(train_loader, 0):
            ## forward pass
            labels, inputs = data['label'].to(device), data['input'].to(device)
            outputs = net(inputs)

            ## backward pass
            optimizer.zero_grad()
            loss = fn_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            ## calculate loss function
            loss_arr += [loss.item()]
            print("training: [Epoch %d/%d] [Batch %d/%d] [Loss: %.3f]"
                % (epoch, args.epochs, batch, len(train_loader), np.mean(loss_arr)))

            ## recording events by tensorboard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch)

        ## validation
        with torch.no_grad():
            net.eval()
            loss_arr = []
            for batch, data in enumerate(train_loader, 0):
                ## forward pass & no backward pass
                labels, inputs = data['label'].to(device), data['input'].to(device)
                outputs = net(inputs)

                ## loss function
                loss = fn_loss(outputs, labels)
                loss_arr += [loss.item()]

                ## print loss
                print("validation: [Epoch %d/%d] [Batch %d/%d] [Loss: %.3f]"
                    % (epoch, args.epochs, batch, len(train_loader), np.mean(loss_arr)))

        ## save model
        torch.save(net.state_dict(), 'output/UNet.pth')

    ## close tensorboard & finish
    writer.close()
    print('Finished Training')

