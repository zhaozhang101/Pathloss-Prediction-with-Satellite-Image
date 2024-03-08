'''
pygetmap:
Download web map by cooridinates
'''

# Longitude 经度
# Latitude   纬度
# Mecator x = y = [-20037508.3427892,20037508.3427892]
# Mecator Latitue = [-85.05112877980659，85.05112877980659]

import math
import os
from math import floor, pi, log, tan, atan, exp
from threading import Thread, Lock
import urllib.request as ur
import PIL.Image as pil
import io
import pandas as pd

import scipy.io
import random

MAP_URLS = {
    # "google": "http://mt0.google.com/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}",
    # "google":"http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
    # "google":"http://mt2.google.cn/vt/lyrs=m@167000000&hl=zh-CN&gl=cn&x=26705&y=14226&z=15&s=Galil",
    # "google": "http://mt0.google.com/vt/lyrs={style}&hl=en&gl=en&x={x}&y={y}&z={z}&s=Galileo0",
    "google": "http://mt0.google.com/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}",

    # "google": "http://kh2.google.com/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}",

    "amap": "http://wprd02.is.autonavi.com/appmaptile?style={style}&x={x}&y={y}&z={z}",
    "tencent_s": "http://p3.map.gtimg.com/sateTiles/{z}/{fx}/{fy}/{x}_{y}.jpg",
    "tencent_m": "http://rt0.map.gtimg.com/tile?z={z}&x={x}&y={y}&styleid=3"}

COUNT = 0
mutex = Lock()


# -----------------GCJ02到WGS84的纠偏与互转,火星---------------------------
def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def delta(lat, lon):
    '''
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0  # a: 卫星椭球坐标投影到平面地图坐标系的投影因子。
    ee = 0.00669342162296594323  # ee: 椭球的偏心率。
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}


def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False


def gcj_to_wgs(gcjLon, gcjLat):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"], gcjLat - d["lat"])


def wgs_to_gcj(wgsLon, wgsLat):
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon)
    return wgsLon + d["lon"], wgsLat + d["lat"]


# --------------------------------------------------------------

# ------------------wgs84与web墨卡托互转-------------------------

# WGS-84经纬度转Web墨卡托
def wgs_to_macator(x, y):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2


# Web墨卡托转经纬度
def mecator_to_wgs(x, y):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return x2, y2


'''
东经为正，西经为负。北纬为正，南纬为负
j经度 w纬度 z缩放比例[0-22] ,对于卫星图并不能取到最大，测试值是20最大，再大会返回404.
山区卫星图可取的z更小，不同地图来源设置不同。
'''


# 根据WGS-84 的经纬度获取谷歌地图中的瓦片坐标
def wgs84_to_tile(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not (isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    # w = 85.0511287798 if w > 85.0511287798 else w
    # w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2 ** z
    x = floor(j * num)
    y = floor(w * num)
    lat_deg, lon_deg = num2deg(x, y, z)
    return x, y, lat_deg, lon_deg


def wgs84_to_tile_float(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not (isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    # w = 85.0511287798 if w > 85.0511287798 else w
    # w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2 ** z
    x = (j * num)
    y = (w * num)
    return x, y


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tileframe_to_mecatorframe(zb):
    # 根据瓦片四角坐标，获得该区域四个角的web墨卡托投影坐标
    inx, iny = zb["LT"]  # left top
    inx2, iny2 = zb["RB"]  # right bottom
    length = 20037508.3427892
    sum = 2 ** zb["z"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right buttom
    # 返回四个角的投影坐标
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res


def tileframe_to_pixframe(zb):
    # 瓦片坐标转化为最终图片的四个角像素的坐标
    out = {}
    width = (zb["RT"][0] - zb["LT"][0] + 1) * 256
    height = (zb["LB"][1] - zb["LT"][1] + 1) * 256
    out["LT"] = (0, 0)
    out["RT"] = (width, 0)
    out["LB"] = (0, -height)
    out["RB"] = (width, -height)
    return out


# def gps84_To_Gcj02( lon,lat):
#     a = 6378245.0
#     ee = 0.00669342162296594323
#     # % if (outOfChina(lat, lon))
#     #     % return null
#
#     dLat = transformLat(lon - 105.0, lat - 35.0)
#     dLon = transformLon(lon - 105.0, lat - 35.0)
#     radLat = lat / 180.0 * pi
#     magic = math.sin(radLat)
#     magic = 1 - ee * magic * magic
#     sqrtMagic = math.sqrt(magic)
#     dLat = (dLat * 180.0)/ ((a * (1 - ee)) / (magic * sqrtMagic) * pi)
#     dLon = (dLon * 180.0) / (a/ sqrtMagic* math.cos(radLat) * pi)
#     mgLat = lat + dLat
#     mgLon = lon + dLon
#     return mgLon,mgLat
#
#
# def transformLat(x, y):
#     a = 6378245.0
#     ee = 0.00669342162296594323
#     ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1* x * y + 0.2 * math.sqrt(abs(x))
#     ret = ret + (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
#     ret = ret + (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
#     ret = ret + (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
#     return ret
#
#
# def transformLon(x, y):
#     ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
#     ret = ret + (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
#     ret = ret + (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
#     ret = ret + (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
#     return ret

# -----------------------------------------------------------


class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, index, count, urls, datas, update):
        # index 表示第几个线程，count 表示线程的总数，urls 代表需要下载url列表，datas代表要返回的数据列表。
        # update 表示每下载一个成功就进行的回调函数。
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count
        self.update = update

    def download(self, url):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.76 Safari/537.36'}
        header = ur.Request(url, headers=HEADERS)
        err = 0
        while (err < 3):
            try:
                data = ur.urlopen(header).read()
            except:
                err += 1
            else:
                return data
        raise Exception("Bad network link.")

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue
            self.datas[i] = self.download(url)
            if mutex.acquire():
                self.update()
                mutex.release()


def geturl(source, x, y, z, style):
    '''
    Get the picture's url for download.
    style:
        m for map
        s for satellite
    source:
        google or amap or tencent
    x y:
        google-style tile coordinate system
    z:
        zoom
    '''
    if source == 'google':
        furl = MAP_URLS["google"].format(x=x, y=y, z=z, style=style)
    elif source == 'amap':
        # for amap 6 is satellite and 7 is map.
        style = 6 if style == 's' else 7
        furl = MAP_URLS["amap"].format(x=x, y=y, z=z, style=style)
    elif source == 'tencent':
        y = 2 ** z - 1 - y
        if style == 's':
            furl = MAP_URLS["tencent_s"].format(
                x=x, y=y, z=z, fx=floor(x / 16), fy=floor(y / 16))
        else:
            furl = MAP_URLS["tencent_m"].format(x=x, y=y, z=z)
    else:
        raise Exception("Unknown Map Source ! ")

    return furl


def downpics(urls, multi=10):
    def makeupdate(s):
        def up():
            global COUNT
            COUNT += 1
            print("\b" * 45, end='')
            print("DownLoading ... [{0}/{1}]".format(COUNT, s), end='')

        return up

    url_len = len(urls)
    datas = [None] * url_len
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas, makeupdate(url_len)) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()

    return datas


def getpic(predata_path, file_name, x1, y1, x2, y2, z, source='google', outfile="MAP_OUT.png", style='s'):
    '''
    依次输入左上角的经度、纬度，右下角的经度、纬度，缩放级别，地图源，输出文件，影像类型（默认为卫星图）
    获取区域内的瓦片并自动拼合图像。返回四个角的瓦片坐标
    '''
    x1, y1 = wgs_to_gcj(x1, y1)
    x2, y2 = wgs_to_gcj(x2, y2)

    pos1x, pos1y, lat_deg1, lon_deg1 = wgs84_to_tile(x1, y1, z)
    pos2x, pos2y, lat_deg2, lon_deg2 = wgs84_to_tile(x2, y2, z)

    lenx = pos2x - pos1x + 2
    leny = pos2y - pos1y + 2

    pos2x = pos1x + lenx
    pos2y = pos1y + leny
    # pos1x=pos1x+1
    # pos1y=pos1y+1

    # lenx = pos2x - pos1x
    # leny = pos2y - pos1y
    lat_deg2, lon_deg2 = num2deg(pos2x - 1, pos2y - 1, z)
    lat_deg1, lon_deg1 = num2deg(pos1x, pos1y, z)

    print("range wgs84:", lat_deg1, lat_deg2, lon_deg1, lon_deg2)
    print("tile id:", pos1x, pos2x - 1, pos1y, pos2y - 1)
    print("Total number：{x} X {y}".format(x=lenx - 1, y=leny - 1))

    data_sample = pd.DataFrame([pos1x, pos2x - 1, pos1y, pos2y - 1])
    # data_sample.to_csv(os.path.join(predata_path, 'Data_Orignal' + '_' +  file_name + '.csv'), encoding='utf_8_sig', index=False)
    data_sample.to_csv(os.path.join(predata_path, 'Satellite_Map_DT_Orignal_boundary' + '_' + file_name + '.csv'),
                       encoding='utf_8_sig',
                       index=False)

    urls = [geturl(source, i, j, z, style) for j in range(pos1y, pos2y) for i in range(pos1x, pos2x)]

    datas = downpics(urls)

    print("\nDownload Finished！ Pics Mergeing......")
    outpic = pil.new('RGB', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        try:
            picio = io.BytesIO(data)
            small_pic = pil.open(picio)

            y, x = i // lenx, i % lenx
            outpic.paste(small_pic, (x * 256, y * 256))
        except IOError:
            print("IOError")

    print('Pics Merged！ Exporting......')
    outpic.save(outfile)
    print('Exported to file！')
    return {"LT": (pos1x, pos1y), "RT": (pos2x, pos1y), "LB": (pos1x, pos2y), "RB": (pos2x, pos2y), "z": z}


def screen_out(zb, name):
    if not zb:
        print("N/A")
        return
    print("坐标形式：", name)
    print("左上：({0:.5f},{1:.5f})".format(*zb['LT']))
    print("右上：({0:.5f},{1:.5f})".format(*zb['RT']))
    print("左下：({0:.5f},{1:.5f})".format(*zb['LB']))
    print("右下：({0:.5f},{1:.5f})".format(*zb['RB']))


def file_out(zb, file, target="keep", output="file"):
    '''
    zh_in  : tile coordinate
    file   : a text file for ArcGis
    target : keep = tile to Geographic coordinate
             gcj  = tile to Geographic coordinate,then wgs84 to gcj
             wgs  = tile to Geographic coordinate,then gcj02 to wgs84
    '''
    pixframe = tileframe_to_pixframe(zb)
    Xframe = tileframe_to_mecatorframe(zb)
    for i in ["LT", "LB", "RT", "RB"]:
        Xframe[i] = mecator_to_wgs(*Xframe[i])
    if target == "keep":
        pass;
    elif target == "gcj":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = wgs_to_gcj(*Xframe[i])
    elif target == "wgs":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        raise Exception("Invalid argument: target.")

    if output == "file":
        f = open(file, "w")
        for i in ["LT", "LB", "RT", "RB"]:
            f.write("{0[0]:.5f}, {0[1].5f}, {1[0].5f}, {1[1].5f}\n".format(pixframe[i], Xframe[i]))
        f.close()
        print("Exported link file to ", file)
    else:
        screen_out(Xframe, target)


def map_loader(predata_path, file_name, lon1, lat1, lon2, lat2, zoom, source, path):
    x = getpic(predata_path, file_name, lon1, lat1, lon2, lat2,
               zoom, source=source, style='s',
               outfile=(path + "_" + str(zoom) + 'm.jpg'))
    return x

# 根据指定的经纬度信息下载卫星图片
if __name__ == '__main__':
    predata_path = 'data'  # 卫星图边界的csv文件的保存位置
    file_name = 'grid'  # csv文件的名字的最后部分
    lon1 = 116.4
    lon2 = 116.5
    lat1 = 40
    lat2 = 39.9
    zoom = 19
    source = 'amap'
    path = 'data1'  # 下载后照片存放的文件名的前面部分
    map_loader(predata_path, file_name, lon1, lat1, lon2, lat2, zoom, source, path)
