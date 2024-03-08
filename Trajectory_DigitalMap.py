from pylab import *
import pandas as pd
import os

def map_clutter(Z,xmin,xmax,ymin,ymax,resolution):#输入地图坐标信息，精度信息和转化好的矩阵，绘制clutter地图并在其中标注基站和测试点
    fig,ax = plt.subplots(2,2)
    ax[0,0].matshow(Z.T)
    X_min,X_max,Y_min,Y_max = [],[],[],[]

    for index_grid in range(len(gridname)):
        UEfilename = os.path.join('Data', cityname[index_city], 'dataset/New_data/measurement_data', UEname[index_grid])
        data_ue = pd.read_csv(UEfilename)
        x_UE = np.array(data_ue["x"][:].tolist())
        y_UE = np.array(data_ue["y"][:].tolist())
        xmi = np.min(x_UE);xma = np.max(x_UE)
        ymi = np.min(y_UE);yma = np.max(y_UE)
        X_min.append(xmi);X_max.append(xma)
        Y_min.append(ymi);Y_max.append(yma)
        x_UE = (x_UE - xmin)/resolution
        y_UE = (ymax - y_UE)/resolution
        ax[0,0].scatter(x_UE,y_UE,marker='o',s=0.3)
    ax[0,0].set_xlim((min(X_min)-xmin)/resolution,(max(X_max)-xmin)/resolution)
    ax[0,0].set_ylim((ymax-min(Y_min))/resolution,(ymax-max(Y_max))/resolution)
    ax[0,0].set_title('clutter_map')
    ax[0,0].axis('off')

    for i in range(len(gridname)):
        BSfilename = os.path.join('Data', cityname[index_city], 'dataset/New_data/BS', 'BS.xlsx')
        data = pd.read_excel(BSfilename)
        x_BS = np.array(data["x"][:].tolist())
        y_BS = np.array(data["y"][:].tolist())
        x_BS = (x_BS - xmin) / resolution
        y_BS = (ymax - y_BS) / resolution
        UEfilename = os.path.join('Data', cityname[index_city], 'dataset/New_data/measurement_data', UEname[i])
        data_ue = pd.read_csv(UEfilename)
        x_UE = np.array(data_ue["x"][:].tolist())
        y_UE = np.array(data_ue["y"][:].tolist())
        x_UE = (x_UE - xmin) / resolution
        y_UE = (ymax - y_UE) / resolution
        fir = (i+1)//2 ;sec = (i+1) % 2
        ax[fir,sec].matshow(Z.T)
        ax[fir,sec].scatter(x_UE, y_UE, marker='o', s=0.4)
        ax[fir,sec].scatter(x_BS, y_BS, marker='*', s=0.9)
        ax[fir,sec].set_xlim((min(X_min) - xmin) / resolution, (max(X_max) - xmin) / resolution)
        ax[fir,sec].set_ylim((ymax - min(Y_min)) / resolution, (ymax - max(Y_max)) / resolution)
        ax[fir,sec].set_title(f'grid:{i}')
        ax[fir,sec].axis('off')
    plt.show()

'''
主程序如下
用于生成clutter地图以及基站接收机轨迹等
'''

#修改城市标签
index_city= 0;index_grid = 0

cityname = ['hz','nb','wz']
if index_city == 1:
    gridname = ['deg+grid14', 'deg+grid15']
else :
    gridname = ['deg+grid14','deg+grid15','deg+grid16']
UEname = ['deg+grid14.csv','deg+grid15.csv','deg+grid16.csv']
clutter = os.path.join('Data',cityname[index_city],'dataset/map/clutter/index.txt')
data_cluter = pd.read_table(clutter, header=None, delim_whitespace=True)
xmin = int(data_cluter[1])
xmax = int(data_cluter[2])
ymin = int(data_cluter[3])
ymax = int(data_cluter[4])
resolution = int(data_cluter[5])
'''
默认二进制转化为十进制的矩阵已经生成
如无，可运行“data_clutter”生成
'''
Z = np.load(f"binainfo_map_{cityname[index_city]}.npy")
map_clutter(Z,xmin,xmax,ymin,ymax,resolution)

