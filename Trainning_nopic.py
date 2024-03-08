import model
from torch.utils.data import DataLoader
import torch
import torch.nn
import torch.cuda
from tools import *
from torch import optim
import time
from torch.backends import cudnn
import sys
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

'''
Trainning_wopic.py:输入距离、角度、小区编号来实现预测
'''


def train(model1, epoc, trainset, lr, weight_decay):  # 用于训练的函数
    picname = os.path.join(savename, f'{key}.jpg')

    train_dataloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=batchsize, drop_last=True)
    optimizer = optim.Adam(model1.parameters(), lr=lr, weight_decay=weight_decay)
    # 动态学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    LOSS_RMSE = []
    LOSS_ME = []
    MSE = torch.nn.MSELoss(reduction='mean')
    since = time.time()
    for epoch in range(epoc):
        model1.train(True)
        counter = 0
        loss_me = 0
        loss_rmse = 0
        for data in train_dataloader:
            label, distance, theta, bs = data
            label, distance, theta, bs = label.float().to(device), distance.float().to(device), theta.float().to(
                device), bs.float().to(device)

            optimizer.zero_grad()
            outputs = model1((distance, theta, bs))

            meloss = torch.mean(label * PL_F - outputs * PL_F)
            mseloss = torch.sqrt(MSE(outputs * PL_F, label * PL_F))

            loss_me = loss_me + meloss * batchsize
            loss_rmse = loss_rmse + mseloss * batchsize
            loss = mseloss
            loss.backward()
            optimizer.step()
            counter += 1

        scheduler.step()
        LOSS_ME_avg = loss_me / (counter * batchsize)
        LOSS_RMSE_avg = loss_rmse / (counter * batchsize)
        LOSS_ME.append(LOSS_ME_avg.item())
        LOSS_RMSE.append(LOSS_RMSE_avg.item())
        draw_curve(epoch, LOSS_ME, LOSS_RMSE, picname)
        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'LOSS_ME in epoch{epoch} is: {LOSS_ME_avg}')
        print(f'LOSS_RMSE in epoch{epoch} is: {LOSS_RMSE_avg ** 0.5}')
    return model1


def test(testset):  # 用于测试的函数

    train_dataloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=batchsize, drop_last=True)
    loss_me = 0
    loss_rmse = 0
    counter = 0
    MSE = torch.nn.MSELoss()
    for data in train_dataloader:
        label, distance, theta, bs = data
        label, distance, theta, bs = label.float().to(device), distance.float().to(device), theta.float().to(
            device), bs.float().to(device)
        outputs = model1((distance, theta, bs))

        meloss = torch.mean(label - outputs)
        rmseloss = torch.sqrt(MSE(label * PL_F, outputs * PL_F))
        loss_me = loss_me + meloss * batchsize
        loss_rmse = loss_rmse + rmseloss * batchsize
        counter = counter + 1
    LOSS_ME_avg = loss_me / (counter * batchsize)
    LOSS_RMSE_avg = loss_rmse / (counter * batchsize)
    print('==========================test========================')
    print(f'RMSELoss is:{LOSS_RMSE_avg ** 0.5}', f'MELoss is:{LOSS_ME_avg}')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    print('Using {} device'.format(device))
    np.random.seed(1)
    torch.manual_seed(1)
    # ===================================================
    key = 'Hangzhou'
    traincsv = './post-Data/training_hz.csv'
    Dataset = CustomImageDataset_wopic(traincsv)
    PL_F = Dataset.pl_s
    trainset, testset = train_test_split(Dataset, test_size=0.2, random_state=42)

    batchsize = 1
    model1 = model.Net1().cuda()
    # model1.load_state_dict(torch.load(f'result/{key}_model_parameters.pth'))
    epoc = 10
    lr = 0.0001
    weight_decay = 1e-4
    # ====================================================
    savename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    logname = os.path.join(savename, f'{key}.txt')
    sys.stdout = Record(logname)

    print('learning rate:', lr, 'weight decay:', weight_decay)
    model1 = train(model1, epoc, trainset, lr, weight_decay)
    model1.eval()
    with torch.no_grad():
        test(testset)

    torch.save(model1.state_dict(), f'result/{key}_model_parameters.pth')

