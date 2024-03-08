import torchvision.models as models
import torch
import torch.nn as nn
import torch.cuda

'''
Net1:不使用卫星图片，使用小区编号、距离和方位角的线性模型
Net2:使用卫星图片、小区编号、距离和方位角的线性模型
'''


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.relu = torch.nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 64)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(128 + 16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, input):
        distance, theta, bs = input
        batchsize = distance.shape[0]
        a = self.layer1(distance)
        b = self.layer2(theta)
        c = self.layer3(bs)
        d = torch.concatenate([a, b, c], dim=0)

        return self.layer4(d)


# 有卫星图片作为输入
class Net2(nn.Module):  # 只用距离d来预测pathloss的模型
    def __init__(self):
        super(Net2, self).__init__()
        self.relu = torch.nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 64)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((32,32)),
            nn.Conv2d(16, 64, kernel_size=(3,3)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.AdaptiveAvgPool2d(1)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(128 + 16 + 64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, input):
        distance, theta, bs, pic = input
        if distance.ndim ==1:
            distance, theta, bs = distance.unsqueeze(1), theta.unsqueeze(1), bs.unsqueeze(1)
        batchsize = distance.shape[0]
        a = self.layer1(distance)
        b = self.layer2(theta)
        c = self.layer3(bs)
        d = self.layer4(pic).squeeze()
        if d.ndim ==1:
            d = d.unsqueeze(0)

        e = torch.concatenate([a, b, c, d], dim=1)
        # e = torch.concatenate([a, b, c], dim=1)

        return self.layer5(e)



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, std=0.001)


class resnetmodel(nn.Module):  # 模型ResNet50魔改
    def __init__(self, batchsize):
        super(resnetmodel, self).__init__()
        self.relu = torch.nn.ReLU()
        self.bs = batchsize
        self.model = models.resnet50(pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.con = nn.Conv2d(1024, 256, kernel_size=1)
        layer1 = []
        layer1 += [nn.Linear(256, 16, bias=False)]
        layer1 = nn.Sequential(*layer1)
        # layer1.apply(weights_init_classifier)    #权重初始化可加，加入后初始状态相对恒定
        self.layer1 = layer1

        layer2 = []
        layer2.append(nn.Linear(1, 8, bias=False))
        layer2.append(nn.BatchNorm1d(8))
        layer2.append(nn.ReLU())
        layer2.append(nn.Linear(8, 16, bias=False))
        layer2.append(nn.BatchNorm1d(16))
        layer2.append(nn.ReLU())
        layer2 = nn.Sequential(*layer2)
        # layer2.apply(weights_init_classifier)
        self.layer2 = layer2

        layer3 = []
        layer3.append(nn.Linear(16, 4, bias=False))
        layer3.append(nn.BatchNorm1d(4))
        layer3.append(nn.ReLU())
        layer3.append(nn.Linear(4, 1, bias=False))
        layer3 = nn.Sequential(*layer3)
        # layer3.apply(weights_init_classifier)
        self.layer3 = layer3

    def forward(self, input):
        x0 = input[0] / 255
        dis = input[1]
        x = self.model.conv1(x0)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.pool(x)
        x = self.con(x)
        x = torch.squeeze(x)
        x = self.layer1(x)
        dis = torch.reshape(dis, (self.bs, 1)).to(torch.float32)
        dis = self.layer2(dis)
        y = x + dis
        y = self.layer3(y)
        return y


class original(nn.Module):  # 复现的matlab中的模型
    def __init__(self, batchsize):
        super(original, self).__init__()
        self.conv_11 = nn.Conv2d(3, 100, kernel_size=7, padding=3)
        self.bn_11 = nn.BatchNorm2d(100)
        self.bs = batchsize
        self.relu = nn.ReLU()
        self.conv_21 = nn.Conv2d(100, 20, kernel_size=3, padding=1)
        self.bn_21 = nn.BatchNorm2d(20)
        self.fc111 = nn.Linear(64 * 64 * 20, 10)
        self.fc21b = nn.Linear(1, 10)
        self.fc21a = nn.Linear(10, 50)
        self.fc22 = nn.Linear(50, 10)
        self.fc_a11 = nn.Linear(10, 10)
        self.fc_a21 = nn.Linear(10, 2)
        self.fc_a31 = nn.Linear(2, 1)

    def forward(self, input):
        x0 = input[0]
        dis = input[1]
        bs = x0.shape[0]
        dis = torch.reshape(dis, (self.bs, 1)).to(torch.float32)
        x = self.conv_11(x0)
        x = self.bn_11(x)
        x = self.relu(x)
        x = self.conv_21(x)
        x = self.bn_21(x)
        x = x.reshape(bs, -1)
        x = self.relu(x)
        x1 = self.fc111(x)
        d = self.fc21b(dis)
        d = self.relu(d)
        d = self.fc21a(d)
        d = self.relu(d)
        x2 = self.fc22(d)
        x = x1 + x2
        x = self.relu(self.fc_a11(x))
        x = self.relu(self.fc_a21(x))
        x = self.fc_a31(x)
        return x
