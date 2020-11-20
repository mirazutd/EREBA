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
from models import base
import argparse
import models_skip as models
from training_model import FlatResNet32
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SkipNet Attack')
    parser.add_argument('mode', choices=['mode1', 'mode2'])
    args = parser.parse_args()
    return args


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
])



transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])




#trainset = torchdata.CIFAR10(root='data/', train=True, download=True, transform=transform_train)
testset = torchdata.CIFAR10(root='data/', train=False, download=True, transform=transform_test)


#train_loader = torchdata2.DataLoader(trainset, batch_size=len(trainset))
#test_loader = torchdata2.DataLoader(testset, batch_size=1,shuffle=False,num_workers=4)
#train_dataset_array = next(iter(train_loader))[0].numpy()
test_loader = torchdata2.DataLoader(testset, batch_size=len(testset),shuffle=False,num_workers=4)
test_dataset_array = next(iter(test_loader))[0].numpy()
#print(test_dataset_array.shape)
'''test_dataset_array = next(iter(test_loader))
test_dataset_array=[t.numpy() for t in test_dataset_array]
print(type(test_dataset_array))
print(np.asarray(test_dataset_array).shape)'''

'''test_dataset_array = testset.data
print(test_dataset_array.shape)'''
#print(test_dataset_array[0][0])
# time.sleep(10)
'''training_arr = np.reshape(test_dataset_array, (10000, 3072))[:5000]
# test_dataset_array=np.reshape(training_arr, (5000,3,32,32))
# print(test_dataset_array[0][0])
time.sleep(5)
test_arr = np.reshape(test_dataset_array, (10000, 3072))[5000:]
lst = []
# print(np.reshape(test_dataset_array, (10000,3072)))
print(training_arr[0])'''
lst=[]
with open('skip_cifar_train_avg.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[1]))

target_arr = np.asarray(lst)[0:8000]
target_arr2 = np.asarray(lst)


def test_model():
    # create model
    model = models.__dict__['cifar10_rnn_gate_110'](False)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load("resnet-110-rnn-sp-cifar10.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    return model






def tanh_rescale(x, x_min=-2.22, x_max=2.5):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)

def loss_op(output, dist, scale_const):

    loss1 =  (- 1 )* output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

layer_config = [18, 18, 18]
net = FlatResNet32(base.BasicBlock, layer_config, num_classes=1)

model = net
model.load_state_dict(torch.load('cifar_train_skip_new2.pth'))
model.eval().cuda()

#x, y = Variable(torch.Tensor(test_dataset_array[8000:0])), Variable(torch.Tensor(target_arr2))

skip_model=test_model().eval().cuda()





def get_blocks(masks):
    skips = [mask.data.le(0.5).float().mean() for mask in masks]
    sum = 0
    #print(skips)
    for s in skips:
        # print(s)
        value = s.tolist()
        if (value == 1):
            sum += 1
    return (54-sum)
x, y = Variable(torch.Tensor(test_dataset_array)), Variable(torch.Tensor(target_arr2))
torch_dataset = Data.TensorDataset(x, y)
BATCH_SIZE = 1
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, )
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

def denormalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = t[:, i, :, :] * std[i] + mean[i]
    return n
def normalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = (t[:, i, :, :] - mean[i]) / std[i]
    return n


def mode1():
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        input_var = Variable(batch_x, volatile=True).cuda()
        output2, masks, _ = skip_model(input_var)
        sm1 = get_blocks(masks)
        if (sm1 <= 44):
            c = torch.tensor(10000, requires_grad=False).cuda()
            w = torch.rand([1, 3, 32, 32], device="cuda", requires_grad=True)
            input_adv=torch.rand([1, 3, 32, 32], device="cuda", requires_grad=True)
            optimizer = optim.Adam([w], lr=0.01)
            for i in range(5000):
                input_adv = normalize((torch.tanh(w) + 1) / 2)
                output = model.forward_single(input_adv)
                loss1 = l2_dist(input_var, input_adv)
                loss = loss_op(output, loss1, c)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            output2, masks, _ = skip_model(input_adv)
            sm2 = get_blocks(masks)
            print("initial no of blocks ", sm1)
            print("no of blocks after EREBA", sm2)
            with open("output_skip" + str(step) + ".obj", "wb") as f:
                pickle.dump(denormalize(input_adv).cpu().tolist(), f)


def mode2():
    N = 30

    for index in range(N):
        w = torch.rand([1, 3, 32, 32], device="cuda", requires_grad=True)
        input_adv = torch.rand([1, 3, 32, 32], device="cuda", requires_grad=True)
        optimizer = optim.Adam([w], lr=0.01)
        for i in range(500):
            input_adv = normalize((torch.tanh(w) + 1) / 2)
            output = model.forward_single(input_adv)
            loss = loss_op(output, torch.tensor(0, device="cuda"), torch.tensor(1, device="cuda"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        output2, masks, _ = skip_model(input_adv)
        sm2 = get_blocks(masks)
        print("no of blocks after EREBA", sm2)




args = parse_args()
print(args)
if(args.mode=="mode1"):
    mode1()
else:
    mode2()