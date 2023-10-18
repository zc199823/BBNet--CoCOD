import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from datetime import datetime
from BBNet import BBNet
#from model.CPD_ResNet_models import CPD_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')#288，384
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models

model = BBNet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

def structure_loss(pred, mask):

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

#------------------train_path--------------------------
image_root = '/media/deep507/4tb/ZC/train/image/'
gt_root = '/media/deep507/4tb/ZC/train/groundtruth/' 

#------------------loss function-----------------------
CE = torch.nn.BCEWithLogitsLoss()


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

#-----------------------------------------------------------------------
        #S_2_pred, S_3_pred, S_4_pred, S_5_pred, S_g_pred = model(images) 可根据返回值调整，性能会有提升
#        loss1 = structure_loss(S_2_pred, gts)
#        loss2 = structure_loss(S_3_pred, gts)
#        loss3 = structure_loss(S_4_pred, gts)
#        loss4 = structure_loss(S_5_pred, gts)
#        loss5 = structure_loss(S_g_pred, gts)
        #loss2 = CE(dets, gts)
#        loss = loss1 + loss2 + loss3 + loss4 + loss5
#--------------------------------------------------------------------------
        s1 = model(images)
        loss1 = CE(s1, gts)
       # loss2 = structure_loss(s1, gts)
       # loss = loss1 + loss2
        loss = loss1 
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

       # if i % 400 == 0 or i == total_step:
        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
              format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

#------------------------train_save---------------------------------
    save_path = './pth/BBNet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path +  'train_w.pth' + '.%d' % epoch)

print("Let's go!")

for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

#houjia
    datasets = os.listdir(image_root)
    for dataset in datasets:
        print(dataset)
        image_root1 = image_root + dataset + '/'
        gt_root1=gt_root+dataset+'/'
       
        train_loader = get_loader(image_root1, gt_root1, batchsize=opt.batchsize, trainsize=opt.trainsize)
        total_step = len(train_loader)

        train(train_loader, model, optimizer, epoch)

















