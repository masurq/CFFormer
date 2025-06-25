import os
import cv2
import shutil
from osgeo import gdal
import numpy as np


class Common:

    def __init__(self):
        pass

    @classmethod
    def check_path(cls, path, reset=False):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if reset == True:
                shutil.rmtree(path)
                os.makedirs(path)

    @classmethod
    def load_mean_file(cls, mean_file):
        mean_value = []
        with open(mean_file, 'r') as f:
            for line in f:
                line = line.strip()
                mean_value.append(float(line))
        return mean_value

    @classmethod
    def load_std_file(cls, std_file):
        std_value = []
        with open(std_file, 'r') as f:
            for line in f:
                line = line.strip()
                std_value.append(float(line))
        return std_value

    @classmethod
    def image_read_cv2(cls, path, mode='RGB'):
        img_BGR = cv2.imread(path).astype('float32')
        assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
        if mode == 'RGB':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        elif mode == 'YCrCb':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        return img

    @classmethod
    def gdal_to_numpy(cls, filename):
        dataset = gdal.Open(filename)
        if dataset is None:
            cls.logger.log('WARNING', 'GDAL can not open {} !'.format(filename), show_time=False, print_type='print')
            return None

        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        img_nbands = dataset.RasterCount

        band_list = [i + 1 for i in range(img_nbands)]
        if img_nbands == 3 or img_nbands == 4:
            band_list[0] = 3
            band_list[2] = 1

        img = np.zeros(1)
        for i in range(img_nbands):
            band = dataset.GetRasterBand(band_list[i])
            img_arr = band.ReadAsArray().reshape((img_height, img_width, 1))

            if i == 0:
                img = img_arr
            else:
                img = np.append(img, img_arr, axis=2)

        return img


common = Common()
