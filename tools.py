import os
import errno
import matplotlib.pyplot as plt
import sys

import numpy
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn as nn
import numpy as np
fig = plt.figure()
ax0 = fig.add_subplot(121, title="me-loss")
ax1 = fig.add_subplot(122, title="rmse-loss")
x_epoch=[]

def draw_curve(epoch,me,rmse,name):          #绘制Loss曲线的函数
    x_epoch.append(epoch)
    ax0.plot(x_epoch, me, 'bo-', label='me_loss')
    ax1.plot(x_epoch, rmse, 'bo-', label='rmse_loss')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(name)

def mkdir_if_missing(dir_path):       #不需care
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Record(object):                        #记录输出结果到txt中的函数
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:  # 如果路径存在，
            mkdir_if_missing(os.path.dirname(path))
            self.file = open(path, 'w')  # 打开路径
        else:
            print('The path provided is wrong!')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CustomImageDataset_wopic(Dataset):                 #数据集中取数据的函数
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        pl = np.array(self.img_labels['pathloss'])
        ds = np.array(self.img_labels['distance'])
        ta = np.array(self.img_labels['theta'])
        bs = np.array(self.img_labels['bs'])
        self.pl_m, self.pl_s = np.mean(pl), np.std(pl)
        self.ds_m, self.ds_s = np.mean(ds), np.std(ds)
        self.ta_m, self.ta_s = np.mean(ta), np.std(ta)
        self.bs_m, self.bs_s = np.mean(bs), np.std(bs)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # label = self.img_labels.iloc[idx, 0]
        # distance = self.img_labels.iloc[idx, 1]
        # theta = self.img_labels.iloc[idx, 2]
        # bs = self.img_labels.iloc[idx, 3]
        label = (self.img_labels.iloc[idx, 0] - self.pl_m) / self.pl_s
        distance = (self.img_labels.iloc[idx, 1] - self.ds_m) / self.ds_s
        theta = (self.img_labels.iloc[idx, 2] - self.ta_m) / self.ta_s
        bs = (self.img_labels.iloc[idx, 3] - self.bs_m) / self.bs_s
        return label, distance, theta, bs

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        pl = np.array(self.img_labels['pathloss'])
        ds = np.array(self.img_labels['distance'])
        ta = np.array(self.img_labels['theta'])
        bs = np.array(self.img_labels['bs'])
        self.pl_m, self.pl_s = np.mean(pl), np.std(pl)
        self.ds_m, self.ds_s = np.mean(ds), np.std(ds)
        self.ta_m, self.ta_s = np.mean(ta), np.std(ta)
        self.bs_m, self.bs_s = np.mean(bs), np.std(bs)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # label = self.img_labels.iloc[idx, 0]
        # distance = self.img_labels.iloc[idx, 1]
        # theta = self.img_labels.iloc[idx, 2]
        # bs = self.img_labels.iloc[idx, 3]
        label = (self.img_labels.iloc[idx, 0] - self.pl_m) / self.pl_s
        distance = (self.img_labels.iloc[idx, 1] - self.ds_m) / self.ds_s
        theta = (self.img_labels.iloc[idx, 2] - self.ta_m) / self.ta_s
        bs = (self.img_labels.iloc[idx, 3] - self.bs_m) / self.bs_s
        image = (np.array(read_image(self.img_labels.iloc[idx, 4]),dtype=numpy.float64)-128)/256

        return label, distance, theta, bs ,image



def weights_init_classifier(m):                 #初始化线性层权重的函数
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, std=0.001)