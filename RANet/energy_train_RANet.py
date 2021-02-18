import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import csv
import torch.utils.data as torchdata2
import numpy as np
from torch.autograd import Variable
from torch import optim
import time
from torch import autograd
import sys
import torch.nn.functional as F
import torch.utils.data as Data
from model import base
from util import Partition
from training_model import FlatResNet32
import math
import torchvision.datasets as datasets

transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                        ])



transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                        ])




#trainset = torchdata.CIFAR10(root='data/', train=True, download=True, transform=transform_test)


normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
train_set = datasets.CIFAR100(root='data/', train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))


train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=len(train_set), shuffle=False,
                num_workers=4, pin_memory=False)
testset = torchdata.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
#train_loader = torchdata2.DataLoader(trainset, batch_size=len(trainset))
#test_loader = torchdata2.DataLoader(testset, batch_size=len(testset))
train_dataset_array = next(iter(train_loader))[0].numpy()
#test_dataset_array = next(iter(test_loader))[0].numpy()
#print(test_dataset_array[0][0])
# time.sleep(10)
'''training_arr = np.reshape(test_dataset_array, (10000, 3072))[:5000]

# test_dataset_array=np.reshape(training_arr, (5000,3,32,32))

# print(test_dataset_array[0][0])

time.sleep(5)
test_arr = np.reshape(test_dataset_array, (10000, 3072))[5000:]

lst = []
# print(np.reshape(test_dataset_array, (10000,3072)))
print(training_arr[0])
'''
lst=[]
with open('RANet_train_energy_avg.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[0]))




target_arr = np.asarray(lst)

#target_arr2 = np.asarray(lst)[8000:]
# target_arr=np.reshape(target_arr,(5000,1))
#target_arr2 = np.asarray(lst)[5000:]
# target_arr2=np.reshape(target_arr2,(5000,1))
# print(target_arr)


#print(train_dataset_array[0:46115])
x, y = Variable(torch.Tensor(train_dataset_array)), Variable(torch.Tensor(target_arr))
'''lst=[]
with open('Cifar_Energy.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[2]))

target_arr2 = np.asarray(lst)'''





def MMSELoss(prediction, b_y):
    f = torch.as_tensor(0.0)
    loss=Variable(f, requires_grad=True).cuda()
    for a, b in zip(prediction, b_y):

        l1=b.item()
        #print(l1)

        #loss = torch.add((a - b) ** 2, loss)
        if(l1>40.0):
            loss=torch.add(torch.mul((a-b)**2,5),loss)

        else:
            loss=torch.add((a - b) ** 2, loss)

    #time.sleep(10)
    return loss

layer_config = [18, 18, 18]
net=rnet = FlatResNet32(base.BasicBlock, layer_config, num_classes=1)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = MMSELoss  # this is for regression mean squared loss

BATCH_SIZE = 200
EPOCH = 200
print(x.size())

print(y.size())
torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, )


print("in")
# start training
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        '''block_vals=[]
        block_vals.append(epoch)
        f = open("ddnn_train.csv", "a")
        writer = csv.writer(f)
        # print(energy)
        writer.writerow(block_vals)
        f.close()'''

        b_x = Variable(batch_x).cuda()
        b_y = Variable(batch_y).cuda()
        #print(b_x.size())
        #print(b_y.size())


        prediction = net.forward_single(b_x)  # input x and predict based on x



        loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)
        #print("loss ",loss)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients



model = net

torch.save(model.state_dict(), 'RANet_train_block3.pth')


'''
input=torch.Tensor(test_dataset_array[8000:0]).cuda()

x = Variable(input).cuda()
predictions=net.forward_single(x)
pred=predictions.tolist()

with open('compare_skip.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(pred[0], target_arr2))'''