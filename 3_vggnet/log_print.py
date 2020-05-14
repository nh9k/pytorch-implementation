from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cfgs = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E'}
    for idx, arch in enumerate(['vgg11','vgg13','vgg16_1','vgg16','vgg19']):
        f = open('result/log_txt/'+ arch + '_log.txt', mode='r', encoding='utf-8')
        line_num = 0
        loss, acc = [], []
        while True:
            line = f.readline()
            if not line:
                break
            if line.find("vgg")!=-1:
                continue
            line_num += 1
            loss.append(float(line[line.find("loss:") + 6:line.find("loss:") + 6 + 5]))
            acc.append(float(line[line.find("acc:") + 5:line.find("acc:") + 5 + 5]))
            #print(epoch, iter, loss, acc)

        plt.figure(1)
        plt.plot(range(line_num), loss, label=(arch+'('+cfgs[idx+1])+')')
        plt.legend(loc='upper left')
        plt.title('Loss/train')

        plt.figure(2)
        plt.plot(range(line_num), acc, label=(arch+'('+cfgs[idx+1])+')')
        plt.legend(loc='upper left')
        plt.title('Accuracy/train')
    plt.show()
    f.close()