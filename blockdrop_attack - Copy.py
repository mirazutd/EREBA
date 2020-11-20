import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.utils as torchvutils
import csv
import torch.utils.data as torchdata2
import numpy as np
from torch import optim
from models import base
from training_model import FlatResNet32
import utils
import argparse
import pickle
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SkipNet Attack')
    parser.add_argument('mode', choices=['mode1', 'mode2'])
    args = parser.parse_args()
    return args

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

HP_T = 100
# HP_C = 1000

BATCH_SIZE = 100
testset = torchdata.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
test_loader = torchdata2.DataLoader(testset, batch_size=BATCH_SIZE)


def get_block_count(agent, inputs):
    with torch.no_grad():
        probs, _ = agent(inputs)

    policy = probs.clone()
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0
    sm = policy.sum(1)
    return sm


def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)


def loss_op(output, dist, scale_const):
    loss1 = (- 1) * output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss


layer_config = [18, 18, 18]
net = FlatResNet32(base.BasicBlock, layer_config, num_classes=1)

model = net
model.load_state_dict(torch.load('cifar_train2.pth'))
model.eval().cuda()

# x, y = Variable(torch.Tensor(test_dataset_array)), Variable(torch.Tensor(target_arr2))

rnet, agent = utils.get_model('R110_C10')

utils.load_checkpoint(rnet, agent, 'cv/finetuned/R110_C10_gamma_10/' \
                      + 'ckpt_E_2000_A_0.936_R_1.95E-01_S_16.93_#_469.t7')
rnet.eval().cuda()
agent.eval().cuda()

saves = [10, 100, 500, 1000, 5000]
results = {}

for k in range(5):
    HP_C = 10 ** (k - 2)
    results[HP_C] = {}
    for j in saves:
        results[HP_C][j] = [0 for i in range(len(testset) // BATCH_SIZE)]

# w = torch.rand([N, 3, 32, 32], device="cuda", requires_grad=True).float()






pbar = tqdm(total=5*(len(testset) // BATCH_SIZE)*5000)
def mode1():
    inputs = []
    outputs = []
    threads = []
    # f1 = open("test_inputs.obj", "wb")
    # f2 = open("test_outputs.obj", "wb")
    for c in range(5):
        HP_C = 10 ** (c - 2)
        for step, (batch_x, batch_y) in enumerate(test_loader):  # for each training step
            input_var = batch_x.cuda()
            sm1 = get_block_count(agent, input_var)
            #print("Batch No: ", step)
            if torch.sum(sm1 <= 10) > 0:
                input_var = input_var[sm1 <= 10, :, :, :]
                bc = input_var.shape[0]
                w = torch.rand([bc, 3, 32, 32], device="cuda", requires_grad=True)
                optimizer = optim.Adam([w], lr=0.01)
                c = torch.tensor(HP_C, requires_grad=False).cuda()
                for i in range(5000):
                    input_adv = (torch.tanh(w) + 1) / 2
                    output = model.forward_single(input_adv)
                    loss1 = l2_dist(input_var, input_adv)
                    loss = loss_op(output, loss1, c)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (i + 1) in saves:
                        results[HP_C][i + 1][step] = input_adv.cpu().tolist()

                print("blocks ",get_block_count(agent, input_adv))
    with open("outputs_blockdrop.obj", "wb") as f:
        pickle.dump(results, f)



def mode2():
    N = 30
    w = torch.rand([N, 3, 32, 32], device="cuda", requires_grad=True)
    optimizer = optim.Adam([w], lr=0.01)
    for i in range(500):
        input_adv = (torch.tanh(w) + 1) / 2
        output = model.forward_single(input_adv)
        loss = loss_op(output, torch.tensor(0, device="cuda"), torch.tensor(1, device="cuda"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("blocks ", get_block_count(agent, input_adv))

args = parse_args()
print(args)
if(args.mode=="mode1"):
    mode1()
else:
    mode2()