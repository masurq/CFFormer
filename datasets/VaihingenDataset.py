from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import torch
from utils.Common import common
from datasets.Augmentor import AugToolkit


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.astype(np.int32)).long()


# normalize_opt = transforms.Normalize([0.31745465668076894,
#                                       0.3215167806711388,
#                                       0.46952500724871127],
#                                      [0.14469553637468746,
#                                       0.15067260693812012,
#                                       0.2106402697703121
#                                       ])
#
# normalize_dsm = transforms.Normalize([0.12416015649414193],
#                                      [0.14496429814418474])
# all_train
normalize_opt = transforms.Normalize([0.31498907610064836,
                                      0.31924508151960196,
                                      0.4695957391912401],
                                     [0.1439061221084104,
                                      0.15065303828126658,
                                      0.2105321008459388])

normalize_dsm = transforms.Normalize([0.12028253133968912],
                                     [0.1433826778967437])

img_opt_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_opt
])

img_dsm_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_dsm
])

mask_transform = MaskToTensor()


class VaihingenDataset(Dataset):
    def __init__(self, class_name, root, mode=None,
                 img_opt_transform=img_opt_transform,
                 img_aux_transform=img_dsm_transform,
                 mask_transform=mask_transform,
                 aug_params=None,
                 ):
        # 数据相关
        self.mode = mode
        self.class_names = class_name
        self.img_opt_transform = img_opt_transform
        self.img_dsm_transform = img_aux_transform
        self.mask_transform = mask_transform
        self.sync_img_mask = []
        self.output_names = []

        img_opt_dir = os.path.join(root, 'NIRRG')
        img_dsm_dir = os.path.join(root, 'dsm')
        mask_dir = os.path.join(root, 'gt_no_boundary')

        for img_filename in os.listdir(mask_dir):
            img_mask_pair = (os.path.join(img_opt_dir, img_filename.replace('.png', '.tif')),
                             os.path.join(img_dsm_dir, img_filename.replace('.png', '.tif')),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
            self.output_names.append(img_filename)

        self.aug_toolkit = None
        if aug_params is not None:
            self.aug_toolkit = AugToolkit(aug_params)

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_opt_path, img_dsm_path, mask_path = self.sync_img_mask[index]
        output_names = self.output_names[index]

        img_opt = common.gdal_to_numpy(img_opt_path)
        img_dsm = common.gdal_to_numpy(img_dsm_path)
        mask = common.gdal_to_numpy(mask_path)

        img = np.append(img_opt, img_dsm, axis=2)
        img = np.append(img, mask, axis=2)

        if self.mode == 'train':
            if self.aug_toolkit is not None:
                img = self.aug_toolkit.run(img)

        mask = img[:, :, -1].copy()
        img_opt = img[:, :, :3].copy()
        img_dsm = img[:, :, 3:-1].copy()

        if self.img_dsm_transform is not None:
            img_opt = self.img_opt_transform(img_opt)
            img_dsm = self.img_dsm_transform(img_dsm)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img_opt, img_dsm, mask, output_names

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names


if __name__ == "__main__":
    pass
