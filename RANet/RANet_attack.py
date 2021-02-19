import csv
import torchvision.datasets as datasets
import os
import torch
import torchvision.transforms as transforms
from torch import optim
from model import base
import torch.utils.data as torchdata2
import argparse
import pickle
# import models_skip as models
from training_model import FlatResNet32
import torchvision.datasets as torchdata
import torchvision.utils as torchvutils
from tqdm import tqdm
from args import args
import models
from dataloader import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SkipNet Attack')
    parser.add_argument('mode', choices=['mode1', 'mode2'])
    args = parser.parse_args()
    return args


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
BATCH_SIZE = 10

normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])

testset=datasets.CIFAR10(root='data/', train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))

val_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=1, shuffle=False,
                num_workers=4, pin_memory=False)
test_loader = val_loader

train_loader, val_loader, test_loader = get_dataloaders(args)

#testset = torchdata.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
#test_loader = torchdata2.DataLoader(testset, batch_size=BATCH_SIZE)
mean = [0.4914, 0.4824, 0.4467]
std = [0.2471, 0.2435, 0.2616]


def normalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = (t[:, i, :, :] - mean[i]) / std[i]
    return n


def partition(t, n_rows=3, n_cols=2):
    # img2=torch.as_tensor(img2)
    BS, C, W, H = t.shape
    parts = []
    part_W, part_H = W // n_cols, H // n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            parts.append(t[:, :, c * part_W:(c + 1) * part_W, r * part_H:(r + 1) * part_H])
    # print(parts)
    # img2 = torch.zeros((1, 6, 3, 16, 10), device='cuda')
    return torch.stack(parts).permute([1, 0, 2, 3, 4])





def test_model():
    model = torch.load('model2.pth')
    return model


def tanh_rescale(x, x_min=-1.7, x_max=2.05):
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
    loss1 = (- 1) * output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss


def denormalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = t[:, i, :, :] * std[i] + mean[i]
    return n


MAX_PROCESS = 600
layer_config = [18, 18, 18]
net = FlatResNet32(base.BasicBlock, layer_config, num_classes=1)
model = net
model.load_state_dict(torch.load('RANet_train_block2.pth'))
model.eval().cuda()
##ddnn_model = test_model().eval().cuda()
saves = [10, 100, 500, 1000, 5000]
results = {}

#if args.gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


RANet_model = getattr(models, 'RANet')(args)
RANet_model = torch.nn.DataParallel(RANet_model.cuda())
state_dict = torch.load('model_best.pth.tar')['state_dict']

#state_dict = torch.load('model_best2.pth.tar')
RANet_model.load_state_dict(state_dict)

RANet_model.eval().cuda()
for k in range(5):
    HP_C = 10 ** k
    results[HP_C] = {}
    for j in saves:
        results[HP_C][j] = [0 for i in range(len(testset) // BATCH_SIZE)]


def mode1(test_loader):
    for step, (batch_x, batch_y) in enumerate(test_loader):  # for each training step
        for c in range(5):
            HP_C = 10 ** c
            count = 0
            input_var = batch_x.cuda()
            ###s = partition(input_var)

            output = RANet_model(input_var)

            print(output)
            if (len(output) == 8):
                continue
            info = []
            info.append(step)
            info.append(HP_C)
            info.append(len(output))
            f = open("RANet_ip_res.csv", "a")
            writer = csv.writer(f)
            # print(energy)
            writer.writerow(info)
            f.close()
            # If index value is 6 then the energy consumption is highest.
            # if torch.sum(index <= 5) > 0:
            # input_var = input_var[index <= 5]
            bc = 1
            w = torch.rand([bc, 3, 32, 32], device="cuda", requires_grad=True)
            input_adv = torch.rand([bc, 3, 32, 32], device="cuda", requires_grad=True)
            optimizer = optim.Adam([w], lr=0.01)
            c = torch.tensor(HP_C, requires_grad=False).cuda()
            count = count + bc
            for i in range(5000):
                input_adv = normalize((torch.tanh(w) + 1) / 2)
                output = model.forward_single(input_adv)
                loss1 = l2_dist(input_var, input_adv)
                loss = loss_op(output, loss1, c)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                '''if (i + 1) in saves:
                    results[HP_C][i + 1][step] = denormalize(input_adv).cpu().tolist()'''
                # print("Starting new thread for saving batch_no: " + str(step) + " C: " + str(HP_C) + \
                #      " Iteration: " + str(i + 1))
                # st = Thread(target=calculateEnergy,
                #             args=(step, partition(input_adv).cpu().tolist(), HP_C, i + 1))
                # st.start()
                # threads.append(st)

            # s = partition(input_adv)
            # predictions, eta, index = ddnn_model(s)
            # print("indexes ", index)
            output = RANet_model(input_adv)
            info = []
            info.append(step)
            info.append(HP_C)
            info.append(len(output))
            f = open("RANet_ip_res2.csv", "a")
            writer = csv.writer(f)
            # print(energy)
            writer.writerow(info)
            f.close()


            # break
    # for t in threads:
    #     t.join()
    # print(results)
    '''with open("outputs_ddnn.obj", "wb") as f:
        pickle.dump(results, f)'''


def mode2():
    for index in range(30):
        N = 1
        w = torch.rand([N, 3, 32, 32], device="cuda", requires_grad=True)
        input_adv = torch.rand([N, 3, 32, 32], device="cuda", requires_grad=True)
        optimizer = optim.Adam([w], lr=0.01)

        output = RANet_model(input_adv)
        print(output)
        '''for o in output:
            print(len(o))'''
        # print("indexes ", len(output))
        for i in range(500):
            input_adv = normalize((torch.tanh(w) + 1) / 2)
            output = model.forward_single(input_adv)
            loss = loss_op(output, torch.tensor(0, device="cuda"), torch.tensor(1, device="cuda"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # s = partition(input_adv)
        # predictions, eta, index = ddnn_model(s)
        output = RANet_model(input_adv)
        print("indexes ", len(output))
        info=[]
        info.append(len(output))
        f = open("RANet_uni_res.csv", "a")
        writer = csv.writer(f)
        # print(energy)
        writer.writerow(info)
        f.close()
        #o = model.forward_single(input_adv)
        #el, ind = torch.max(o, 0)
    '''with open("outputs_ddnn_uni.obj", "wb") as f:
        pickle.dump(denormalize(input_adv[ind, :, :, :]).cpu().tolist(), f)'''
    # return denormalize(input_adv[ind, :, :, :]). cpu().tolist()
    # calculateEnergy(0, partition(input_adv[ind, :, :, :]).cpu().tolist(), 1, 10)
    # probs, _ = agent(input_adv[ind, :, :, :])
    # probs[probs < 0.5] = 0.0
    # probs[probs >= 0.5] = 1.0
    # torchvutils.save_image(input_adv[ind, :, :, :], "blockdrop_mode2.png")
    # plt.imsave("blockdrop_mode2.png", input_adv[ind, :, :, :].squeeze(0).permute([1, 2, 0]).cpu().tolist())
    # print(torch.sum(probs))


'''args = parse_args()
print(args)
if (args.mode == "mode1"):
    mode1()
else:
    mode2()'''


mode1(test_loader)
#mode2()