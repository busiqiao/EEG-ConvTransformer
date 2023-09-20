# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/15 21:09
 @name: 
 @desc:
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_load.dataset import EEGImagesDataset
from model.conv_transformer import *
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from utils import train, test, learning_rate_scheduler
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号

torch.manual_seed(1234)
np.random.seed(1234)

batch_size = 64
learning_rate = 0.0001
decay = 0.170
gamma = 0.6
epochs = 40
k = 10
exp_id = 'debug'
history = np.zeros((k, epochs))

dataset = EEGImagesDataset(path='H:/EEG/EEGDATA/img_pkl_124')
k_fold = KFold(n_splits=k, shuffle=True)
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('显卡数量：', torch.cuda.device_count(), '  ', torch.cuda.get_device_name(0), '  显卡号：',
              torch.cuda.current_device())
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    global_step = 0
    for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):
        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3,
                                  prefetch_factor=2)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
                                  prefetch_factor=1)
        n_t = len(train_ids)
        n_v = len(valid_loader)
        print('Fold -', fold, ' num of train and test: ', n_t, n_v)

        model = ConvTransformer2(num_classes=6, channels=72, num_heads=12, E=432, F=768, T=32, depth=2).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
        summary = SummaryWriter(log_dir='./log/' + exp_id + '/' + str(fold) + '_fold/')

        for epoch in range(epochs):
            for step, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                lr = learning_rate_scheduler(epoch=epoch, lr=learning_rate, gamma=gamma)
                loss, y_ = train(model=model, optimizer=optimizer, x=x, y=y, lr=lr)
                global_step += 1
                if step % 50 == 0:
                    corrects = (torch.argmax(y_, dim=1).data == y.data)
                    acc = corrects.cpu().int().sum().numpy() / batch_size
                    summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                    summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)
                    print('epoch:{}/{} step:{}/{} global_step:{} lr:{:.8f} loss={:.5f} acc={:.3f}'.format(
                        epoch, epochs, step, int(n_t / batch_size), global_step, lr, loss, acc))
                    # print(meminfo.used/1024/1024**2, 'G')  #已用显存大小
            print('Training done')
            sum_acc = 0
            for step, (x, y) in enumerate(valid_loader):
                if batch_size == 64 and step == 81:  # 跳过第81个step的原因是kfold分配的验证集在batich_size=64时，
                    continue  # 第81个step无法填满，导致除以精度异常甚至报错
                loss, acc = test(model=model, x=x, y=y)
                acc = acc / batch_size
                sum_acc += acc
                summary.add_scalar(tag='ValLoss', scalar_value=loss, global_step=global_step)
                summary.add_scalar(tag='ValAcc', scalar_value=acc, global_step=global_step)
                print('test step:{}/{} loss={:.5f} acc={:.3f}'.format(step, int(n_v / batch_size), loss, acc))
            print('本次epoch测试精度：{:.5f}'.format(sum_acc / n_v))
            history[fold, epochs] = sum_acc / n_v
            print('Testing done')
        av_fold_acc = np.sum(history[fold]) / epochs
        print('本次fold平均精度：{:.5f}'.format(av_fold_acc))
    av_acc = np.sum(history) / k
    print('平均准确率：{:.5f}'.format(av_acc))
