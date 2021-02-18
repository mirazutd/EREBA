import os
import math
import time
import shutil
import models
import time
from dataloader import get_dataloaders
from args import args
from adaptive_inference import dynamic_evaluate
from op_counter import measure_model
from tx2_predict import PowerLogger,getNodes
import csv
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

#torch.manual_seed(args.seed)

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model = getattr(models, 'RANet')(args)
model = torch.nn.DataParallel(model.cuda())
criterion = nn.CrossEntropyLoss().cuda()

train_loader, val_loader, test_loader = get_dataloaders(args)
#state_dict = torch.load('model_best.pth.tar')['state_dict']

state_dict = torch.load('model_best2.pth.tar')
model.load_state_dict(state_dict)




def validate(val_loader, model, criterion):
    #batch_time = AverageMeter()
    #losses = AverageMeter()
    #data_time = AverageMeter()
    top1, top5 = [], []


    model.eval()

    #end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)

            print(output)
            #data_time.update(time.time() - end)
            info=[]
            '''for index in range(5):
                pl = PowerLogger(interval=0.05, nodes=list(filter(lambda n: n[0].startswith('module/'), getNodes())))
                pl.start()

                output = model(input_var)

                pl.stop()
                eng, tm = pl.getTotalEnergy()
                info.append(eng)
            info.append(len(output))

            f = open("RANet_train_energy2.csv", "a")
            writer = csv.writer(f)
            # print(energy)
            writer.writerow(info)
            f.close()'''

            #output = model(input_var)
            #print(output)

            #time.sleep(5)


validate(train_loader, model, criterion)