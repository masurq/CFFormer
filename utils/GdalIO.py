import numpy as np
from osgeo import gdal


class GdalIO:

    def __init__(self):
        pass

    @classmethod
    def read(cls, path):
        dataset = gdal.Open(path)
        if dataset is None:
            print('WARNING: [GDAL] can not open %s !' % path)
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

    @classmethod
    def write(cls, path, img):
        img = img.reshape((1, img.shape[0], img.shape[1]))
        datatype = gdal.GDT_Byte
        im_bands, im_height, im_width = img.shape

        band_list = [i + 1 for i in range(im_bands)]
        if im_bands == 3 or im_bands == 4:
            band_list[0] = 3
            band_list[2] = 1

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW'])
        for i in range(im_bands):
            dataset.GetRasterBand(band_list[i]).WriteArray(img[i])
        del dataset
