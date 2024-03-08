import torch
import math
import numpy as np
import pandas as pd
import os
from ctypes import *
import time

pi=math.pi
aa = 6378245.0
ee = 0.00669342162296594323

def wgs84_to_tile(j, w, z): #转化为瓦片坐标
    j = (j + 180) / 360
    w = np.log(np.tan((90 + w) * pi / 360)) / (pi / 180)
    w = w / 180
    w = 1 - (w + 1) / 2
    num = 2 ** z
    x = j * num
    y = w * num
    lat_deg, lon_deg=num2deg(x, y, z)
    return x, y,lat_deg, lon_deg

def num2deg(xtile,ytile,zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180
    lat_rad = np.arctan(np.sinh(pi * (1 - 2 * ytile / n)))
    lat_deg = np.degrees(lat_rad)
    return lat_deg, lon_deg

def gps84_to_Gcj02(lon,lat):    #转化为火星坐标系

    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * pi
    magic = np.sin(radLat)
    magic = 1 - np.power(magic, 2)*ee
    sqrtMagic = np.power(magic,0.5)
    dLat = (dLat * 180.0) / ((aa * (1 - ee)) / (magic * sqrtMagic) * pi)
    dLon = (dLon * 180.0) / (aa / sqrtMagic * np.cos(radLat) * pi)
    mgLat = lat + dLat
    mgLon = lon + dLon
    return mgLon,mgLat

def transformLat(x,y):       #经度转换算法

    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * (abs(x))**0.5
    ret = ret + (20.0 * np.sin(6.0 * x * pi) + 20.0 * np.sin(2.0 * x * pi)) * 2.0 / 3.0
    ret = ret + (20.0 * np.sin(y * pi) + 40.0 * np.sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret = ret + (160.0 * np.sin(y / 12.0 * pi) + 320 * np.sin(y * pi / 30.0)) * 2.0 / 3.0
    return ret

def transformLon(x,y):        #纬度转换算法
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * (abs(x))**0.5
    ret = ret + (20.0 * np.sin(6.0 * x * pi) + 20.0 * np.sin(2.0 * x * pi)) * 2.0 / 3.0
    ret = ret + (20.0 * np.sin(x * pi) + 40.0 * np.sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret = ret + (150.0 * np.sin(x / 12.0 * pi) + 300.0 * np.sin(x / 30.0 * pi)) * 2.0 / 3.0
    return ret

def resolutionMapTile(lat,z): #有关测量精度的
    lat = lat/180*pi
    resolution=np.cos(lat)*2*pi*6378137/256/2**z
    return resolution

def normal(a):                #算array中数的平方和的开方
    sum = 0
    for i,item in enumerate(a):
        sum += item**2
    return sum**0.5

def GridMap(file_clutter, cityname):     #根据clutter里的index.txt提取相关信息，到相应的二进制文件中读取
    data_cluter = pd.read_table(file_clutter, header=None, delim_whitespace=True)
    file_domob = str(data_cluter[0])[5:]
    for i, item in enumerate(file_domob):
        if item == '\n':
            file_domob = file_domob[:i]
            file_domob = ''.join(file_domob)
            break
    xmin = data_cluter[1]
    xmax = data_cluter[2]
    ymin = data_cluter[3]
    ymax = data_cluter[4]
    resolution = data_cluter[5]
    Xsize = int((xmax - xmin) / resolution)
    Ysize = int((ymax - ymin) / resolution)
    file = os.path.join('Data', cityname, 'dataset/map/clutter', file_domob)
    read_domob(file, Xsize, Ysize, cityname)

def converge_num(sn,sn0):            #数据类型uint16，所以一次读取两个字节，排除len=9的b'00\r'和b'00\n'的特殊字符，其他转化为十进制。
    if len(sn) == 9:
        return 0
    else :
        data1 = int(sn[9], 16);data2 = int(sn[8], 16)
        data3 = int(sn[5], 16);data4 = int(sn[4], 16)
        data = int(data1 + data2 * 16 + data3 * 16 ** 2 + data4 * 16 ** 3)
        if data == sn0:
            return 0
        else :
            return data

def read_domob(file, Xsize, Ysize, cityname):        #读取二进制地图信息并转化为矩阵
    since = time.time()
    index = 0
    i = 0 ; j = 0 ; sn0 = 0
    b = np.zeros(shape=(Xsize, Ysize), dtype=c_uint16)
    with open(file, 'rb') as f:
        while index < Xsize * Ysize:
            sn = str(f.read(2))
            if index == 0:
                data1 = int(sn[9], 16);data2 = int(sn[8], 16)
                data3 = int(sn[5], 16);data4 = int(sn[4], 16)
                data = data1 + data2 * 16 + data3 * 16 ** 2 + data4 * 16 ** 3
                sn0 = int(data)
            data = converge_num(sn,sn0)
            b[i, j] = data
            if (i + 1) % Xsize == 0:
                j = j + 1; i = 0
            else:
                i = i + 1
            index = index + 1
    np.save(f'binainfo_map_{cityname}', b)
    time_elapsed = time.time() - since
    print('Conversion complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


'''
将clutter里的二进制地图信息转化好(转化为一个np.array矩阵)，并存到程序文件目录中的.npy文件中;
切分电子地图
'''

if __name__ == '__main__':
    # 选择城市
    index_city = 1
    cityname = ['hz', 'nb', 'wz']
    file_clutter = os.path.join('Data', cityname[index_city], 'dataset/map/clutter/index.txt')

    '''
    主程序
    '''
    GridMap(file_clutter, cityname[index_city])





