from torch.autograd import Variable
import vggnet
from utils import *
import argparse
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_names = sorted(name for name in vggnet.__dict__
                         if name.startswith("vgg") and not name.endswith("net"))
                         #and callable(vggnet.__dict__[name])) #호출 가능 여부

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                        choices=model_names,
                        help='a model for training,  default: vgg19')
    args = parser.parse_args()


    trainloader, _ , _ = build_datasets()
    net = vggnet.__dict__[args.arch]()
    net.to(device) #net.cuda() #오류
    print(net)
    f = open(args.arch + '_log.txt', mode='wt', encoding='utf-8')
    f.write(args.arch + '\n')
    writer = SummaryWriter(comment=args.arch)

    criterion, optimizer = Optimizer(net)

    for epoch in range(150):  # 데이터셋을 수차례 반복합니다.
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            # 입력을 받은 후,
            inputs, labels = data
            # Variable로 감싸고
            inputs, labels = inputs.to(device), labels.to(device)
            #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #오류

            # 변화도 매개변수를 0으로 만든 후
            optimizer.zero_grad()

            # 학습 + 역전파 + 최적화
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 모델의 가중치 파라미터
            # params = list(net.parameters())
            # # print(len(params))
            # print('Before')
            # print(params[0])  # conv1's .weight
            #
            # # 모델의 state_dict 출력
            # print("Model's state_dict:")
            # for param_tensor in net.state_dict():
            #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
            #
            # # 옵티마이저의 state_dict 출력
            # print("Optimizer's state_dict:")
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])

            def fn_acc(output, label):
                prediction = torch.softmax(output, dim=1)
                return ((prediction.max(dim=1)[1]==label).type(torch.float)).mean()

            acc = fn_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            # 모델의 가중치 파라미터
            # for p in net.parameters():
            #     print(p.requires_grad)
            # params = list(net.parameters())
            # # print(len(params))
            # print('after optimizer step')
            # print(params[0])  # conv1's .weight

            # 통계 출력
            running_loss += loss.item()  #loss.data[0] 오류
            running_acc += acc.item()
            # print('[%5d] loss: %.3f' % (i + 1, loss.item()))
            if i % 46 == 45:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f, acc: %.3f' %
                      (epoch + 1, i / 46 + 1, running_loss / 46, running_acc / 46))
                f.write('[%d, %5d] loss: %.3f, acc: %.3f\n' %
                      (epoch + 1, i / 46 + 1, running_loss / 46, running_acc / 46))
                writer.add_scalar('Loss/train', running_loss / 46, epoch * len(trainloader) + i)
                writer.add_scalar('Accuracy/train', running_acc / 46, epoch * len(trainloader) + i)

                running_loss = 0.0
                running_acc = 0.0
    writer.close()
    f.close()
    print('Finished Training')
    PATH = './' + args.arch + '.pth'
    torch.save(net.state_dict(), PATH)

# https://pytorch.org/docs/stable/notes/serialization.html
    # The first(recommended) saves and loads only the model parameters:
    #
    # torch.save(the_model.state_dict(), PATH)
    #
    # Then later:
    #
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))
    #
    # The second saves and loads the entire model:
    #
    # torch.save(the_model, PATH)
    #
    # Then later:
    #
    # the_model = torch.load(PATH)