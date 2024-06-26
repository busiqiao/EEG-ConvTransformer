import os
import random
import numpy as np
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import model.modal_parament
from data_load.dataset import EEGImagesDataset
from model.conv_transformer import *
from torch.utils.tensorboard import SummaryWriter
from utils.util import train, test, select_class, save_result
from torchinfo import summary

modal_variant = ['CT-Slim', 'CT-Fit', 'CT-Wide']
classes_name = ['6-Category', '72-Exemplar', 'HF-IO', 'HF-Exemplar', 'IO-Exemplar']

# 根据需要选择训练modal、类别、日志路径，只需要修改这里！！！
classes = '72-Exemplar'
models = 'CT-Slim'
exp_id = 'test'

# 初始化固定变量
variant = modal_variant.index(models)
num_class = select_class(classes=classes)
selected_modal = model.modal_parament.select_modal(num_class=num_class, variant=variant)
batch_size = 64
learning_rate = 0.0001
k = 10
history = np.zeros((10, 10))
save_path = f'./outputs/{classes}/{models}/'
if os.path.exists(save_path) is False:
    os.makedirs(save_path)

# 固定随机种子
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子
kf = KFold(n_splits=k, shuffle=True, random_state=seed_value)

if __name__ == '__main__':
    dataPath = '/data/EEG72-IMG/'

    print(
        '\r参数设置: num_class={}，epochs={}，batch_size={}，k_fold={}，manual_seed={}, learning_rate={}, decay={}, gamma={}, '
        'exp_id={}'.format(selected_modal.num_class, selected_modal.epochs, batch_size, k,
                           selected_modal.seed_value, learning_rate, selected_modal.decay, selected_modal.gamma,
                           exp_id))

    for i in range(k):
        dataset = EEGImagesDataset(file_path=dataPath, s=i, num_class=num_class)

        for fold, (train_ids, valid_ids) in enumerate(kf.split(dataset)):
            train_sampler = SubsetRandomSampler(train_ids)
            valid_sampler = SubsetRandomSampler(valid_ids)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3,
                                      prefetch_factor=2, drop_last=True)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
                                     prefetch_factor=1, drop_last=True)
            n_t = len(train_loader) * batch_size
            n_v = len(test_loader) * batch_size

            model = ConvTransformer(num_classes=selected_modal.num_class, channels=selected_modal.channels,
                                    num_heads=selected_modal.num_heads, E=selected_modal.E, F=selected_modal.F,
                                    T=selected_modal.T, depth=selected_modal.depth).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=selected_modal.decay)
            scheduler = StepLR(optimizer, step_size=5, gamma=selected_modal.gamma)

            if fold == 0:
                print('\r第{}位受试者:  train_num={}, test_num={}'.format(int(i + 1), n_t, n_v))
            if i == 0 and fold == 0:
                summary(model, input_size=(64, 1, 32, 32, 32))
            summary = SummaryWriter(log_dir=f'./log/{exp_id}/{classes}/{models}/{str(fold)}_fold/')

            global_step = 0
            for epoch in range(selected_modal.epochs):
                # 15个epoch后，每5个epoch调整一次学习率
                if epoch > 10:
                    scheduler.step()

                train_loop = tqdm(train_loader, total=len(train_loader))
                for (x, y) in train_loop:
                    x = x.cuda()
                    y = y.cuda()
                    loss, y_ = train(model=model, optimizer=optimizer, x=x, y=y)
                    global_step += 1
                    corrects = (torch.argmax(y_, dim=1).data == y.data)
                    acc = corrects.cpu().int().sum().numpy() / batch_size
                    summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                    summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)
                    lr = scheduler.get_last_lr()[0]

                    train_loop.set_description(f'Epoch [{epoch + 1}/{selected_modal.epochs}] - Train')
                    train_loop.set_postfix(loss=loss.item(), acc=acc, lr=lr)

                val_loss = None
                val_acc = None
                for (xx, yy) in test_loader:
                    val_loss, acc = test(model=model, x=xx, y=yy)
                    val_acc = acc / batch_size
                avg_val_loss = val_loss.item()
                avg_val_acc = np.mean(val_acc)
                summary.add_scalar(tag='TestLoss', scalar_value=avg_val_loss, global_step=global_step)
                summary.add_scalar(tag='TestAcc', scalar_value=avg_val_acc, global_step=global_step)

            losses = []
            accuracy = []
            test_loop = tqdm(test_loader, total=len(test_loader))
            for (xx, yy) in test_loop:
                val_loss, val_acc = test(model=model, x=xx, y=yy)
                val_acc = val_acc / batch_size
                losses.append(val_loss)
                accuracy.append(val_acc)

                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                test_loop.set_description(f'                Test ')
                test_loop.set_postfix(loss=val_loss.item(), acc=val_acc, lr=current_lr)

            avg_test_acc = np.sum(accuracy) / len(accuracy)
            history[i][fold] = avg_test_acc
            print('\r受试者{}，第{}折测试准确率：{}'.format(i + 1, fold + 1, history[i][fold]))
            print('\r---------------------------------------------------------')

        print(history[i])
        print('\r受试者{}训练完成，平均准确率：{}'.format(i + 1, np.mean(history[i], axis=0)))
        print('\r*************************************************************')

    print(history)
    print('\r训练完成，{}类平均准确率：{}'.format(num_class, np.mean(history)))

    save_result(history=history, save_path=save_path + 'results', classes=classes, dropout=0.5, batch_size=batch_size,
                epochs=selected_modal.epochs)
