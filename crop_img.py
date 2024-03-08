import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import coordinate_conv
import pandas as pd
from PIL import Image
from pylab import *
import os

Image.MAX_IMAGE_PIXELS = None
'''
裁剪照片的主程序
'''
###################################################################################################################################
# 设置1：选择城市
z = 18                             #卫星图精度   马来z==17
index_city= 2; index_grid = 0       #修改index_city来改变城市，注意还需更改gridname等（名字和grid个数也不一致）。index_grid 改变卫星图
###################################################################################################################################

cityname = ['hz','nb','wz','ml']
if index_city == 1 or index_city == 3:
    gridname = ['deg+grid14', 'deg+grid15']
else :
    gridname = ['deg+grid14','deg+grid15','deg+grid16']
UEname = ['deg+grid14.csv','deg+grid15.csv','deg+grid16.csv']
Satell_pic = ['Satellite_Map_DT_Orignal_deg+grid14_18m.jpg','Satellite_Map_DT_Orignal_deg+grid15_18m.jpg',
             'Satellite_Map_DT_Orignal_deg+grid16_18m.jpg']
Satell_grid = ['Satellite_Map_DT_Orignal_boundary_deg+grid14.csv','Satellite_Map_DT_Orignal_boundary_deg+grid15.csv',
              'Satellite_Map_DT_Orignal_boundary_deg+grid16.csv']

#文件路径
imagefolder = os.path.join('Data',cityname[index_city],'train')
UEfilename = os.path.join('Data',cityname[index_city],'dataset/New_data/measurement_data',UEname[index_grid])
BSfilename = os.path.join('Data',cityname[index_city],'dataset/New_data/BS','BS.xlsx')
satellite_map = os.path.join('Data',cityname[index_city],'dataset/data_preprocessing',Satell_grid[index_grid])
satellite_pic = os.path.join('Data',cityname[index_city],'dataset/data_preprocessing',Satell_pic[index_grid])

#读取接收机信息文件
data_ue = pd.read_csv(UEfilename)
lon_UE_hz = np.array(data_ue["lon"][:].tolist())
lat_UE_hz = np.array(data_ue["lat"][:].tolist())

#读取基站信息文件
data = pd.read_excel(BSfilename)
lon_BS_hz = np.array(data["lon"][:].tolist())
lat_BS_hz = np.array(data["lat"][:].tolist())

# 读取卫星图
data = pd.read_csv(satellite_map)
xmin_Smap = data["0"][0];xmax_Smap = data["0"][1];ymin_Smap = data["0"][2];ymax_Smap = data["0"][3]
if index_city == 3:
    z = 17
    xcw, ycw, lat_deg, lon_deg = coordinate_conv.wgs84_to_tile(lon_UE_hz, lat_UE_hz, z)
    xBS, yBS, lat_degBS, lon_degBS = coordinate_conv.wgs84_to_tile(lon_BS_hz, lat_BS_hz, z)
else:
    mgLon, mgLat = coordinate_conv.gps84_to_Gcj02(lon_UE_hz ,lat_UE_hz)              #坐标转换：gps到火星坐标系，再到瓦片
    mgLonBS, mgLatBS = coordinate_conv.gps84_to_Gcj02(lon_BS_hz,lat_BS_hz)
    xcw, ycw, lat_deg, lon_deg = coordinate_conv.wgs84_to_tile(mgLon, mgLat, z)
    xBS, yBS, lat_degBS, lon_degBS = coordinate_conv.wgs84_to_tile(mgLonBS, mgLatBS, z)

# reshape变为两维的，方便拼接操作
xcw = xcw.reshape(1,len(xcw));ycw = ycw.reshape(1,len(ycw))
lat_deg = lat_deg.reshape(1,len(lat_deg));lon_deg = lon_deg.reshape(1,len(lon_deg))
xBS = xBS.reshape(1,len(xBS));yBS = yBS.reshape(1,len(yBS))
lat_degBS = lat_degBS.reshape(1,len(lat_degBS));lon_degBS = lon_degBS.reshape(1,len(lon_degBS))

# 相对于整张卫星图的像素差计算
pixel_UE1= np.floor((xcw-xmin_Smap)*256)+1;pixel_UE2 =np.floor((ycw-ymin_Smap)*256)+1
pixel_BS1=np.floor((xBS-xmin_Smap)*256)+1; pixel_BS2 =np.floor((yBS-ymin_Smap)*256)+1

# 横纵坐标信息拼接
pixel_UE = np.r_[pixel_UE1,pixel_UE2]
pixel_BS = np.r_[pixel_BS1,pixel_BS2]

#创建照片的存放位置
if not os.path.isdir(imagefolder):
    os.mkdir(imagefolder)

img = Image.open(satellite_pic)

# 对一个城市的grid中的接收机位置做循环
for index in range(len(pixel_UE[1])):

    bs = pixel_BS[:,data_ue["cell name"][index]-1]
    #bs是第index个接收机对应的基站的坐标（像素值）
    centrpoint = np.round((bs+pixel_UE[:,index])/2)
    #windowsize1是第一次裁剪的边长的1/2
    windowsize1 = max(np.round(np.max(np.abs(bs-pixel_UE[:,index]))/2)+1,3)*2
    #distance是两者之间的像素距离的1/2 * 1.2倍，也是第二次裁剪的边长的1/2
    distance = np.round(coordinate_conv.normal(bs-pixel_UE[:,index])/2 * 1.2)
    Image1 = img.crop((centrpoint[0] - windowsize1,centrpoint[1] - windowsize1,centrpoint[0]+ windowsize1,
                          centrpoint[1] + windowsize1))

    RelativeVector = bs - centrpoint
    RelativeVector = RelativeVector / coordinate_conv.normal(RelativeVector)
    refVector = [-1, 0]
    RotAngle = np.degrees(np.arccos(np.matmul(RelativeVector, refVector)))
    if RelativeVector[1] > 0:
        RotAngle = -RotAngle
    Image2 = Image1.rotate(RotAngle)
    Image3 = Image2.crop((windowsize1 + 1 - distance,windowsize1 + 1 - distance,windowsize1 + distance,
                          windowsize1 + distance))
    newImage = Image3.resize([64, 64]).convert('L')                                     #改变尺寸为64*64的大小

    ###################################################################################################################################
    # 设置2：保存路径
    newImage.save(os.path.join('Image',f'{cityname[index_city]}',f"{gridname[index_grid]}_{index}.png"))
    #修改以设置图片文件的保存位置。
    ###################################################################################################################################
