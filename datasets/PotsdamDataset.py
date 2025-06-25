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


# RGB
# normalize_opt = transforms.Normalize([0.3335395403566105,
#                                       0.3600049708290334,
#                                       0.33667565271184496],
#                                      [0.14223397537986573,
#                                       0.13681329275440487,
#                                       0.1400808354485504])
#
# normalize_dsm = transforms.Normalize([0.1838982778523105],
#                                      [0.21210445796377778])

# # all_train
# normalize_opt = transforms.Normalize([0.33337263411954154,
#                                       0.3599685295585867,
#                                       0.337120880003789],
#                                      [0.14262200150444931,
#                                       0.1370489226154767,
#                                       0.13984182520738078])

# normalize_dsm = transforms.Normalize([0.18230883000852513],
#                                      [0.20971399266053195 ])

# # RGBIR
# normalize_opt = transforms.Normalize([0.33667565271184496,
#                                       0.3600049708290334,
#                                       0.3335395403566105,
#                                       0.37979488314986815],
#                                      [0.1400808354485504,
#                                       0.13681329275440487,
#                                       0.14223397537986573,
#                                       0.1403097382566462])
#
# normalize_dsm = transforms.Normalize([0.1838982778523105],
#                                      [0.21210445796377778])
# all_train
normalize_opt = transforms.Normalize([0.337120880003789,
                                      0.3599685295585867,
                                      0.33337263411954154,
                                      0.3820126308610401],
                                     [0.13984182520738078,
                                      0.1370489226154767,
                                      0.14262200150444931,
                                      0.13944875042381694])

normalize_dsm = transforms.Normalize([0.18230883000852513],
                                     [0.20971399266053195])

img_opt_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_opt
])

img_dsm_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_dsm
])

mask_transform = MaskToTensor()


class PotsdamDataset(Dataset):
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

        img_opt_dir = os.path.join(root, 'RGBIR')
        img_dsm_dir = os.path.join(root, 'dsm_last')
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
        img_opt = img[:, :, :4].copy()
        img_dsm = img[:, :, 4:-1].copy()

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
