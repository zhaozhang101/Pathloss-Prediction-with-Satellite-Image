import coordinate_conv
import pandas as pd
from PIL import Image
from pylab import *
import os
Image.MAX_IMAGE_PIXELS = None

# 修改城市标签:0、1、2、3 各代表杭州、宁波、温州、马来西亚；
index_city= 3;index_grid = 0
cityname = ['hz','nb','wz','ml']    #宁波因为图片太大(100M)加载出错- 16G以上内存可以加载

if index_city == 2 or index_city == 3:
    gridname = ['deg+grid14', 'deg+grid15']
else :
    gridname = ['deg+grid14','deg+grid15','deg+grid16']

UEname = ['deg+grid14.csv','deg+grid15.csv','deg+grid16.csv']
Satell_pic = ['Satellite_Map_DT_Orignal_deg+grid14_18m.jpg','Satellite_Map_DT_Orignal_deg+grid15_18m.jpg',
             'Satellite_Map_DT_Orignal_deg+grid16_18m.jpg']
Satell_grid = ['Satellite_Map_DT_Orignal_boundary_deg+grid14.csv','Satellite_Map_DT_Orignal_boundary_deg+grid15.csv',
              'Satellite_Map_DT_Orignal_boundary_deg+grid16.csv']

def convege(lon_UE_hz ,lat_UE_hz,xmin_Smap,ymin_Smap):    #输入经纬度信息，以及卫星地图的坐标信息，输出测试点在地图上的像素坐标
    if index_city == 3:
        z = 17
        xcw, ycw, lat_deg, lon_deg = coordinate_conv.wgs84_to_tile(lon_UE_hz, lat_UE_hz, z)
    else:
        z = 18
        mgLon, mgLat = coordinate_conv.gps84_to_Gcj02(lon_UE_hz, lat_UE_hz)
        xcw, ycw, lat_deg, lon_deg = coordinate_conv.wgs84_to_tile(mgLon, mgLat, z)
    xcw = xcw[np.newaxis, :];ycw = ycw[np.newaxis, :]
    xcw = xcw[np.newaxis, :];ycw = ycw[np.newaxis, :]
    pixel_UE1 = np.floor((xcw - xmin_Smap) * 256) + 1
    pixel_UE2 = np.floor((ycw - ymin_Smap) * 256) + 1
    return pixel_UE1,pixel_UE2

fig,ax = plt.subplots(1,len(gridname))
for i in range(len(gridname)):                           #将一个城市的不同的grid显示在一个figure的不同subplot中
    satellite_pic = os.path.join('Data', cityname[index_city], 'dataset/data_preprocessing', Satell_pic[i])
    UEfilename = os.path.join('Data', cityname[index_city], 'dataset/New_data/measurement_data', UEname[i])
    satellite_map = os.path.join('Data', cityname[index_city], 'dataset/data_preprocessing', Satell_grid[i])
    img = Image.open(satellite_pic)
    data_ue = pd.read_csv(UEfilename)
    data = pd.read_csv(satellite_map)
    xmin_Smap = data["0"][0];xmax_Smap = data["0"][1]
    ymin_Smap = data["0"][2];ymax_Smap = data["0"][3]
    lon_UE_hz = np.array(data_ue["lon"][:].tolist())
    lat_UE_hz = np.array(data_ue["lat"][:].tolist())
    x,y = convege(lon_UE_hz, lat_UE_hz,xmin_Smap,ymin_Smap)
    ax[i].imshow(img)
    ax[i].scatter(x,y,marker='o',c='y',s=5)
    ax[i].axis('off')
    ax[i].set_title(f'traject_ue_gid{i}.png"')

plt.show()
