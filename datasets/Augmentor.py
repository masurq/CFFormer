import cv2
import skimage
import random
import numpy as np
import math


class AugToolkit:

    def __init__(self, params=None):
        self.augmentor = Augmentor()

        self.flip = False
        self.rotate = False
        self.crop = False
        self.crop_size = None
        self.hsv = False
        self.hsv_limits = None
        self.hsv_band_list = None
        self.shift_scale_rotate = False
        self.ssr_limits = None
        self.brightness = False
        self.bright_limit = None
        self.noise = False
        self.noise_mode = None
        self.blur = False
        self.blur_mode = None
        self.blur_limit = None
        self.cutout = False
        # self.n_holes = None
        # self.cut_size = None
        if params:
            if 'flip' in params:
                self.flip = params['flip']
            if 'rotate' in params:
                self.rotate = params['rotate']
            if 'crop' in params:
                self.crop = params['crop']
            if 'crop_size' in params:
                self.crop_size = params['crop_size']
            if 'hsv' in params:
                self.hsv = params['hsv']
            if 'hsv_limits' in params:
                self.hsv_limits = params['hsv_limits']
            if 'shift_scale_rotate' in params:
                self.shift_scale_rotate = params['shift_scale_rotate']
            if 'ssr_limits' in params:
                self.ssr_limits = params['ssr_limits']
            if 'brightness' in params:
                self.brightness = params['brightness']
            if 'bright_limit' in params:
                self.bright_limit = params['bright_limit']
            if 'noise' in params:
                self.noise = params['noise']
            if 'noise_mode' in params:
                self.noise_mode = params['noise_mode']
            if 'blur' in params:
                self.blur = params['blur']
            if 'blur_mode' in params:
                self.blur_mode = params['blur_mode']
            if 'blur_limit' in params:
                self.blur_limit = params['blur_limit']
            if 'cutout' in params:
                self.cutout = params['cutout']
            # if 'n_holes' in params:
            #     self.n_holes = params['n_holes']
            # if 'cut_size' in params:
            #     self.cut_size = params['cut_size']

    def run(self, img_src):
        img_dst = img_src
        random.seed()

        if self.flip and self.rotate:
            p_switch_rnd = random.random()
            if p_switch_rnd <= 0.5:
                img_dst = self.augmentor.RandomHorizontalFlip(img_dst)
                img_dst = self.augmentor.RandomVerticalFlip(img_dst)
            else:
                img_dst = self.augmentor.RandomRotate90(img_dst)
        else:
            if self.flip:
                img_dst = self.augmentor.RandomHorizontalFlip(img_dst)
                img_dst = self.augmentor.RandomVerticalFlip(img_dst)
            elif self.rotate:
                img_dst = self.augmentor.RandomRotate90(img_dst)

        if self.crop:
            img_dst = self.augmentor.RandomCrop(img_dst, self.crop_size)

        if self.brightness:
            img_dst = self.augmentor.RandomBrightness(img_dst, self.bright_limit)

        if self.hsv:
            img_dst = self.augmentor.RandomHueSaturationValue(img_dst, self.hsv_limits[0], self.hsv_limits[1],
                                                              self.hsv_limits[2])

        if self.shift_scale_rotate:
            img_dst = self.augmentor.RandomShiftScaleRotate(img_dst, self.ssr_limits[0], self.ssr_limits[1],
                                                            self.ssr_limits[2], self.ssr_limits[3])

        if self.noise:
            img_dst = self.augmentor.RandomNoise(img_dst, self.noise_mode)

        if self.blur:
            img_dst = self.augmentor.RandomBlur(img_dst, self.blur_mode, self.blur_limit)

        if self.cutout:
            # img_dst = self.augmentor.cutout(img_dst, self.n_holes, self.cut_size)
            img_dst = self.augmentor.cutout(img_dst)
        return img_dst


class Augmentor:

    def __init__(self):
        self.aug_utils = AugUtils()

    @classmethod
    def RandomHorizontalFlip(cls, img_src, p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            img_dst = cv2.flip(img_src, 1)
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomVerticalFlip(cls, img_src, p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            img_dst = cv2.flip(img_src, 0)
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomHorizontalVerticalFlip(cls, img_src, p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            img_dst = cv2.flip(img_src, -1)
        else:
            img_dst = img_src
        return img_dst

    # @classmethod
    # def cutout(cls, img_src, n_holes=1, cut_size=(16, 16), f_value=0, p=0.5):
    #     random.seed()
    #     p_rnd = random.random()
    #     if p_rnd <= p:
    #         img_dst = img_src.copy()
    #         h, w = img_dst.shape[0], img_dst.shape[1]
    #         img_dst_p = img_dst[:, :, :-1]
    #         img_dst_label = img_dst[:, :, -1][:, :, np.newaxis]
    #         for _ in range(n_holes):
    #             top = random.randint(0, h - cut_size[0])
    #             left = random.randint(0, w - cut_size[1])
    #             img_dst_p[top:(top + cut_size[0]), left:(left + cut_size[1]), :] = f_value
    #         img_dst = np.append(img_dst_p, img_dst_label, axis=2)
    #
    #     else:
    #         img_dst = img_src
    #
    #     return img_dst

    def cutout(self, img_src, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            img_dst = img_src.copy()
            left, top, h, w = self.aug_utils.cutout_get_params(img_dst, scale, ratio)

            c = random.randint(*value)
            img_dst_p = img_dst[:, :, :-1]
            img_dst_label = img_dst[:, :, -1][:, :, np.newaxis]
            img_dst_p[top:top + h, left:left + w, :] = c
            img_dst = np.append(img_dst_p, img_dst_label, axis=2)

        else:
            img_dst = img_src

        return img_dst

    def RandomRotate90(self, img_src, p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            p_rot_rnd = random.random()
            if p_rot_rnd <= 0.33:
                img_dst = self.aug_utils.Rot90(img_src, 1)
            elif p_rot_rnd > 0.33 and p_rot_rnd < 0.67:
                img_dst = self.aug_utils.Rot90(img_src, 2)
            elif p_rot_rnd >= 0.67:
                img_dst = self.aug_utils.Rot90(img_src, 3)
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomCrop(cls, img_src, crop_size=None):
        if crop_size is None:
            return img_src
        crop_height, crop_width = crop_size
        if crop_height < 0 or crop_width < 0:
            return img_src
        img_src_height = img_src.shape[0]
        img_src_width = img_src.shape[1]
        if crop_height > img_src_height or crop_width > img_src_width:
            return img_src
        random.seed()
        upleft_height = random.randint(0, (img_src_height - crop_height))
        upleft_width = random.randint(0, (img_src_width - crop_width))
        img_dst = img_src[upleft_height:(upleft_height + crop_height), upleft_width:(upleft_width + crop_width), :]
        return img_dst

    @classmethod
    def RandomHueSaturationValue(cls, img_src, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255),
                                 val_shift_limit=(-255, 255), p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            band_count = img_src.shape[2] - 1
            band_group = []
            for i in range(0, band_count, 3):
                band_list = [i + ii for ii in range(3)]
                if band_list[2] + 1 > band_count:
                    band_list = [ii - (band_list[2] + 1 - band_count) for ii in band_list]
                band_group.append(band_list)
            band_group.reverse()

            img_dst = img_src.copy()
            for i in range(len(band_group)):
                img_bgr = img_src[:, :, band_group[i][0]:(band_group[i][2] + 1)]
                img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img_hsv)
                hue_shift = np.int32(np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1))
                h = np.uint8((np.int32(h) + hue_shift) % 180)
                sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1] + 1)
                s = cv2.add(s, sat_shift)
                val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1] + 1)
                v = cv2.add(v, val_shift)
                img_hsv = cv2.merge((h, s, v))
                img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                for ii in range(3):
                    img_dst[:, :, band_group[i][ii]] = img_bgr[:, :, ii]
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomShiftScaleRotate(cls, img_src, shift_limit=(-1.0, 1.0), scale_limit=(-1.0, 1.0),
                               rotate_limit=(-1.0, 1.0), aspect_limit=(-1.0, 1.0), p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            height = img_src.shape[0]
            width = img_src.shape[1]
            channel = img_src.shape[2]

            angle = random.uniform(rotate_limit[0], rotate_limit[1])
            # scale = np.random.uniform(scale_limit[0] + 1, scale_limit[1] + 1)
            scale = random.choice(scale_limit)
            aspect = random.uniform(aspect_limit[0] + 1, aspect_limit[1] + 1)
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([
                [cc, -ss],
                [ss, cc]
            ])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = np.float32(box0)
            box1 = np.float32(box1)
            mat = cv2.getPerspectiveTransform(box0, box1)

            band_count = img_src.shape[2] - 1
            band_group = []
            for i in range(0, band_count, 3):
                band_list = [i + ii for ii in range(3)]
                if band_list[2] + 1 > band_count:
                    band_list = [ii - (band_list[2] + 1 - band_count) for ii in band_list]
                band_group.append(band_list)
            band_group.reverse()

            img_dst = img_src.copy()
            for i in range(len(band_group)):
                img_src_image = img_src[:, :, :-1][:, :, band_group[i][0]:(band_group[i][2] + 1)]
                img_dst_image = cv2.warpPerspective(img_src_image, mat, (width, height), flags=cv2.INTER_CUBIC,
                                                    borderMode=cv2.BORDER_REFLECT_101)
                for ii in range(3):
                    img_dst[:, :, band_group[i][ii]] = img_dst_image[:, :, ii]
            img_src_label = img_src[:, :, -1:]
            img_dst_label = cv2.warpPerspective(img_src_label, mat, (width, height), flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_REFLECT_101)
            img_dst[:, :, -1:] = img_dst_label[:, :, None]
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomBrightness(cls, img_src, bright_limit=(-255, 255), p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            bright_shift = np.random.randint(bright_limit[0], bright_limit[1] + 1)

            band_count = img_src.shape[2] - 1
            band_group = []
            for i in range(0, band_count, 3):
                band_list = [i + ii for ii in range(3)]
                if band_list[2] + 1 > band_count:
                    band_list = [ii - (band_list[2] + 1 - band_count) for ii in band_list]
                band_group.append(band_list)
            band_group.reverse()

            img_dst = img_src.copy()
            for i in range(len(band_group)):
                img_src_image = img_src[:, :, :-1][:, :, band_group[i][0]:(band_group[i][2] + 1)]
                img_dst_image = cv2.add(img_src_image, bright_shift)
                for ii in range(3):
                    img_dst[:, :, band_group[i][ii]] = img_dst_image[:, :, ii]
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomNoise(cls, img_src, mode='gaussian', p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            band_count = img_src.shape[2] - 1
            band_group = []
            for i in range(0, band_count, 3):
                band_list = [i + ii for ii in range(3)]
                if band_list[2] + 1 > band_count:
                    band_list = [ii - (band_list[2] + 1 - band_count) for ii in band_list]
                band_group.append(band_list)
            band_group.reverse()

            img_dst = img_src.copy()
            for i in range(len(band_group)):
                img_src_image = img_src[:, :, :-1][:, :, band_group[i][0]:(band_group[i][2] + 1)]
                img_dst_image = cv2.cvtColor(img_src_image, cv2.COLOR_BGR2RGB)
                img_dst_image = skimage.util.random_noise(img_dst_image, mode=mode)
                img_dst_image = np.uint8(img_dst_image * 255)
                img_dst_image = cv2.cvtColor(img_dst_image, cv2.COLOR_RGB2BGR)
                for ii in range(3):
                    img_dst[:, :, band_group[i][ii]] = img_dst_image[:, :, ii]
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def RandomBlur(cls, img_src, mode=None, blur_limit=7, p=0.5):
        random.seed()
        p_rnd = random.random()
        if p_rnd <= p:
            ksize = np.random.randint(3, blur_limit + 1)

            band_count = img_src.shape[2] - 1
            band_group = []
            for i in range(0, band_count, 3):
                band_list = [i + ii for ii in range(3)]
                if band_list[2] + 1 > band_count:
                    band_list = [ii - (band_list[2] + 1 - band_count) for ii in band_list]
                band_group.append(band_list)
            band_group.reverse()

            img_dst = img_src.copy()
            for i in range(len(band_group)):
                img_src_image = img_src[:, :, :-1][:, :, band_group[i][0]:(band_group[i][2] + 1)]
                if mode is not None:
                    if ksize % 2 == 0:
                        ksize -= 1
                if mode == 'Median':
                    img_dst_image = cv2.medianBlur(img_src_image, ksize=ksize)
                elif mode == 'Gaussian':
                    img_dst_image = cv2.GaussianBlur(img_src_image, ksize=(ksize, ksize), sigmaX=0)
                else:
                    img_dst_image = cv2.blur(img_src_image, ksize=(ksize, ksize))
                for ii in range(3):
                    img_dst[:, :, band_group[i][ii]] = img_dst_image[:, :, ii]
        else:
            img_dst = img_src
        return img_dst

    @classmethod
    def Upsample(cls, img_src, upsample_factor=8):
        img_dst_image = cv2.resize(img_src[:, :, :-1], None, fx=upsample_factor, fy=upsample_factor,
                                   interpolation=cv2.INTER_CUBIC)
        img_dst_label = cv2.resize(img_src[:, :, -1:], None, fx=upsample_factor, fy=upsample_factor,
                                   interpolation=cv2.INTER_NEAREST)
        img_dst_label = img_dst_label[:, :, None]
        img_dst = np.append(img_dst_image, img_dst_label, axis=2)
        return img_dst


class AugUtils:

    def __init__(self):
        pass

    @classmethod
    def Rotate(cls, img_src, degree):
        grad = math.pi * degree / 180.0
        img_src_height = img_src.shape[0]
        img_src_width = img_src.shape[1]
        center_x = (img_src_width + 1) // 2 - 1
        center_y = (img_src_height + 1) // 2 - 1
        img_dst_height = int(img_src_height * math.fabs(math.cos(grad)) + img_src_width * math.fabs(math.sin(grad)))
        img_dst_width = int(img_src_height * math.fabs(math.sin(grad)) + img_src_width * math.fabs(math.cos(grad)))
        mat_rotate = cv2.getRotationMatrix2D((center_x, center_y), degree, 1.0)
        mat_rotate[0][2] += (img_dst_width - img_src_width) / 2.0
        mat_rotate[1][2] += (img_dst_height - img_src_height) / 2.0
        img_dst_image = cv2.warpAffine(img_src[:, :, :-1], mat_rotate, (img_dst_width, img_dst_height),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        img_dst_label = cv2.warpAffine(img_src[:, :, -1:], mat_rotate, (img_dst_width, img_dst_height),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        img_dst = np.append(img_dst_image, img_dst_label, axis=2)
        return img_dst

    @classmethod
    def Rot90(cls, img_src, times=1):
        img_dst = img_src
        for _ in range(times):
            img_dst = np.rot90(img_dst)
        return img_dst

    @classmethod
    def cutout_get_params(cls, img_src, scale, ratio):
        img_h, img_w, img_c = img_src.shape

        s = random.uniform(*scale)
        r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)

        return left, top, h, w
