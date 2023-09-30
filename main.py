import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_load.dataset import EEGImagesDataset
from model.conv_transformer import *
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from utils import train, test
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号

torch.manual_seed(0)
np.random.seed(0)

batch_size = 64
learning_rate = 0.0001
decay = 0.135
gamma = 0.5
epochs = 35
k = 10
exp_id = 'debug'
history = np.zeros(k)

dataset = EEGImagesDataset(path='/home/zdd/Desktop/Projects/data_eeg_visual/img_pkl_124')
k_fold = KFold(n_splits=k, shuffle=True)


def acc_test(test_model, valid, g_step=0):
    flag = 0
    sum_acc = 0
    for s, (xx, yy) in enumerate(valid):
        if batch_size == 64 and s == 81:  # 跳过第81个step的原因是kfold分配的验证集在batich_size=64时，
            continue  # 第81个step无法填满，导致除以精度异常甚至报错
        val_loss, val_acc = test(model=test_model, x=xx, y=yy)
        val_acc = val_acc / batch_size
        sum_acc += val_acc
        flag += 1
        summary.add_scalar(tag='ValLoss', scalar_value=val_loss, global_step=g_step)
        summary.add_scalar(tag='ValAcc', scalar_value=val_acc, global_step=g_step)
        print('test step:{}/{} loss={:.5f} acc={:.3f}'.format(s, int(n_v / batch_size), val_loss, val_acc))

    av_fold_acc = sum_acc / flag
    return av_fold_acc


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
        n_v = len(valid_ids)
        print('Fold -', fold, ' num of train and test: ', n_t, n_v)

        model = ConvTransformer(num_classes=6, channels=8, num_heads=4, E=16, F=256, T=32, depth=2).cuda()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
        summary = SummaryWriter(log_dir='./log/' + exp_id + '/' + str(fold) + '_fold/')

        max_acc = 0
        for epoch in range(epochs):
            if epoch > 10:
                scheduler.step()
            for step, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                loss, y_ = train(model=model, optimizer=optimizer, x=x, y=y)
                global_step += 1
                corrects = (torch.argmax(y_, dim=1).data == y.data)
                acc = corrects.cpu().int().sum().numpy() / batch_size
                summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)
                lr = scheduler.get_last_lr()[0]
                if step % 50 == 0:
                    print('epoch:{}/{} step:{}/{} global_step:{} lr:{:.8f} loss={:.5f} acc={:.3f}'.format(
                        epoch, epochs - 1, step, int(n_t / batch_size), global_step, lr, loss, acc))
            epoch_acc = acc_test(test_model=model, valid=valid_loader, g_step=epoch)
            print('本次epoch测试精度：{:.5f}'.format(epoch_acc))
            if epoch_acc > max_acc:
                max_acc = epoch_acc
        print('Fold {} train done'.format(fold))
        print('本次fold平均精度：{:.5f}'.format(max_acc))
        history[fold] = max_acc
        print('Fold {} test done'.format(fold))
    print(history)
    av_acc = np.sum(history) / k
    print('平均准确率：{:.5f}'.format(av_acc))
