import cv2
import tifffile
import numpy as np
import pandas as pd
from osgeo import gdal


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


def read_tiff(input_file):
    dataset = gdal.Open(input_file)        # 打开一个16位TIFF

    width   = dataset.RasterXSize          # 宽
    height  = dataset.RasterYSize          # 高
    channel = dataset.RasterCount          # 波段数

    geo     = dataset.GetGeoTransform()    # 几何信息
    proj    = dataset.GetProjection()      # 投影

    np_data = np.zeros((height, width, channel)) #这里长和宽是反的

    for i in range(channel):
        band = dataset.GetRasterBand(i + 1)    # 读取每个波段的图片
        np_data[:, :, i] = band.ReadAsArray()  # 四个波段存到一个np张量中

    return np_data, geo, proj


def write_tiff(output_file, np_data, geo, proj, sj):
    Driver = gdal.GetDriverByName("Gtiff")

    height, width, channel = np_data.shape
    dataset = Driver.Create(output_file, width, height, channel, sj)   #gdal.GDT_Byte意思是8位数据
    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)

    for i in range(channel):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(np_data[:, :, i])




def conv16to8(origin16_file, output8_file):
    in16_data, geo, proj = read_tiff(origin16_file)
    height, width, channel = in16_data.shape

    out8_data = np.zeros_like(in16_data)
    for i in range(channel):
        band_max = np.max(in16_data[:, :, i])
        band_min = np.min(in16_data[:, :, i])

        cutmin, cutmax = cumulativehistogram(in16_data[:, :, i], height, width, band_min, band_max)
        compress_scale = (cutmax - cutmin) / 255

        temp = np.array(in16_data[:, :, i])
        temp[temp > cutmax] = cutmax
        temp[temp < cutmin] = cutmin
        out8_data[:, :, i] = (temp - cutmin) / compress_scale

    write_tiff(output8_file, out8_data[:,:,[3,2,1]], geo, proj, gdal.GDT_Byte)


def cumulativehistogram(array_data, rows, cols, band_min, band_max):
    gray_level = int(band_max - band_min + 1)
    gray_array = np.ones(gray_level)

    b = array_data - band_min
    c = np.array(b).reshape(1, -1)
    d = pd.DataFrame(c[0])[0].value_counts()
    for i in range(len(d.index)):
        gray_array[int(d.index[i])] = int(d.values[i])

    counts = rows * cols
    count_percent2  = counts * 0.02
    count_percent98 = counts * 0.98


    cutmin, cutmax = 0, 0
    for i in range(1, gray_level):
        gray_array[i] += gray_array[i - 1]
        if (gray_array[i] >= count_percent2 and gray_array[i - 1] <= count_percent2):
            cutmin = i + band_min

        if (gray_array[i] >= count_percent98 and gray_array[i - 1] <= count_percent98):
            cutmax = i + band_min

    return cutmin, cutmax


conv16to8('ms4.tif', 'ms3.tif')

msf_np   = np.load('msf_f6.npy')
np_data, geo, proj = read_tiff('ms4.tif')
print(np.max(msf_np), np.min(msf_np))
write_tiff('msf_f4.tif', 10000*msf_np, geo, proj, gdal.GDT_UInt16)
np_data, geo, proj = read_tiff('msf_f4.tif')
print(np.max(np_data), np.min(np_data))
conv16to8('msf_f4.tif', 'msf_f3.tif')





