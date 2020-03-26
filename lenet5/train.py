from torch.autograd import Variable
from lenet5 import *
from utils import *

if __name__ == '__main__':

    trainloader, _ , _ = build_datasets()
    net = lenet()
    criterion, optimizer = Optimizer(net)

    for epoch in range(2):  # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 입력을 받은 후,
            inputs, labels = data

            # Variable로 감싸고
            inputs, labels = Variable(inputs), Variable(labels)

            # 변화도 매개변수를 0으로 만든 후
            optimizer.zero_grad()

            # 학습 + 역전파 + 최적화
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 출력
            running_loss += loss.item()  #loss.data[0] 오류
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = './cifar_net.pth'
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