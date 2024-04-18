import os

import torch
import numpy as np


def train(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y)
    loss.backward()
    optimizer.step()
    return loss, y_


def test(model, x, y):
    x = x.cuda()
    y = y.cuda()
    model.eval()
    y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y)
    corrects = (torch.argmax(y_, dim=1).data == y.data)
    acc = corrects.cpu().int().sum().numpy()
    return loss, acc


def save_result(history, save_path, classes, dropout, batch_size, epochs):
    avg_acc = np.mean(history)
    std_acc = np.std(history)
    title = 'classes   \tdropout   \tbatch_size\tepochs    \n'
    parameters = f'{classes:<10}\t{dropout:<10}\t{batch_size:<10}\t{epochs:<10}\n'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, f'{classes}.txt')
    sub_avg = np.round(np.mean(history, axis=1), decimals=4)  # 计算行平均值

    # Check if file already exists
    count = 1
    while os.path.exists(filename):
        filename = os.path.join(save_path, f'{classes}_{count}.txt')
        count += 1

    # Write to file
    with open(filename, 'w') as file:
        # Save transposed numpy array to file
        file.write('All Results\n')
        file.write('sub1\tsub2\tsub3\tsub4\tsub5\tsub6\tsub7\tsub8\tsub9\tsub10\n')
        np.savetxt(file, np.round(history.T, decimals=4), delimiter='\t', fmt='%0.4f')
        file.write('\n')
        file.write(f'Sub Average:\n')
        np.savetxt(file, [sub_avg], delimiter='\t', fmt='%0.4f')  # 将行平均值保存到文件中
        file.write('\n')
        file.write(f'Parameters:\n')
        file.write(title)
        file.write(parameters)
        file.write('\n')
        file.write(f'Average Accuracy:\t{avg_acc:.4f}±{std_acc:.4f}\n')


def select_class(classes='6-Category'):
    if classes == '6-Category':
        n_classes = 6
    elif classes == '72-Exemplar':
        n_classes = 72
    elif classes == 'HF-IO':
        n_classes = 2
    elif classes == 'HF-Exemplar' or 'IO-Exemplar':
        n_classes = 12
    else:
        raise ValueError("错误的分类任务！")
    return n_classes
