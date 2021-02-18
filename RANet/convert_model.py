import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models
import os

from args import args


if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model = getattr(models, 'RANet')(args)
model = torch.nn.DataParallel(model.cuda())
criterion = nn.CrossEntropyLoss().cuda()

#train_loader, val_loader, test_loader = get_dataloaders(args)
state_dict = torch.load('model_best.pth.tar')['state_dict']
model.load_state_dict(state_dict)

torch.save(model.state_dict(), 'model_best2.pth.tar', _use_new_zipfile_serialization=False)
