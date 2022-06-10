from osgeo import gdal
import os
import glob
import numpy as np
import cv2
import pandas as pd
import datetime

def unsplit(pan, size):
    w, h, c = pan.shape
    pan_re = np.zeros([2*w, 2*h])
    for i in range(size):
        for j in range(size):
            pan_re[i::size,j::size] = pan[:, :, 2*i+j]
    return pan_re

def read_tiff(input_file):
    dataset = gdal.Open(input_file)
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize

    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    couts = dataset.RasterCount
    if couts == 1:
       band = dataset.GetRasterBand(1)
       im_data = dataset.ReadAsArray(0, 0, cols, rows)

    return im_data, geo, proj, rows, cols, couts


def compress(origin_16, output_8):
    array_data, geo, proj, rows, cols, couts = read_tiff(origin_16)
    band_max = np.max(array_data)
    band_min = np.min(array_data)
    cutmin, cutmax = cumulativehistogram(array_data, rows, cols, band_min, band_max)
    compress_scale = (cutmax - cutmin) / 255
    array_data[array_data<cutmin]=cutmin
    array_data[array_data>cutmax]=cutmax
    compress_data = (array_data - cutmin) / compress_scale
    merge_img = cv2.merge([compress_data, compress_data, compress_data])
    cv2.imwrite(output_8, merge_img)


def write_tiff(output_file, np_data, geo, proj, sj):
    Driver = gdal.GetDriverByName("Gtiff")
    print(np_data.shape)
    height, width = np_data.shape
    dataset = Driver.Create(output_file, width, height, 1, sj)  
    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)

    band = dataset.GetRasterBand(1)
    band.WriteArray(np_data[:, :])


def cumulativehistogram(array_data, rows, cols, band_min, band_max):

    gray_level = int(band_max - band_min + 1)
    gray_array = np.zeros(gray_level)

    counts = 0
    b = array_data - band_min
    c = np.array(b).reshape(1, -1)
    d = pd.DataFrame(c[0])[0].value_counts()
    for i in range(len(d.index)):
        gray_array[int(d.index[i])] = int(d.values[i])
    counts = rows * cols

    count_percent2 = counts * 0.02
    count_percent98 = counts * 0.98

    cutmax = 0
    cutmin = 0

    for i in range(1, gray_level):
        gray_array[i] += gray_array[i - 1]
        if (gray_array[i] >= count_percent2 and gray_array[i - 1] <= count_percent2):
            cutmin = i + band_min
        if (gray_array[i] >= count_percent98 and gray_array[i - 1] <= count_percent98):
            cutmax = i + band_min

    return cutmin, cutmax


if __name__ == '__main__':
    pan_np   = np.load('pan_f6.npy') 
    #pan_np[:,:,[0,1,2,3]] = pan_np[:,:,[0,1,3,2]]
    pan_np  = unsplit(pan_np, 2)
    np_data, geo, proj, rows, cols, couts = read_tiff('pan.tif')
    print(np.max(pan_np), np.min(pan_np))
    print(np.max(np_data), np.min(np_data))
    write_tiff('pan_f4_qg.tif', 10000*pan_np, geo, proj, gdal.GDT_UInt16)
    np_data, geo, proj, rows, cols, couts = read_tiff('pan_f4_qg.tif')
    print(np.max(np_data), np.min(np_data))
    compress('pan_f4_qg.tif', 'pan_f3_qg.tif')

