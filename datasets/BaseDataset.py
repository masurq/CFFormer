from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import torch
from datasets.Augmentor import AugToolkit
from utils.Common import common


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.astype(np.int32)).long()


normalize = transforms.Normalize([0.4170945191702441, 0.38526318591876313, 0.3159387510870505, 0.3923401028603987],
                                 [0.051841639563110116, 0.05823086604198551, 0.06877715681327681,
                                  0.10486000210642858])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
    # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

mask_transform = MaskToTensor()


class BaseDataset(Dataset):
    def __init__(self, class_name, root, mode=None,
                 img_transform=img_transform,
                 mask_transform=mask_transform,
                 aug_params=None,
                 ):
        # 数据相关
        self.mode = mode
        self.class_names = class_name
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sync_img_mask = []
        self.output_names = []

        img_dir = os.path.join(root, 'img')
        mask_dir = os.path.join(root, 'lbl')

        for img_filename in os.listdir(mask_dir):
            img_mask_pair = (os.path.join(img_dir, img_filename),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
            self.output_names.append(img_filename)

        self.aug_toolkit = None
        if aug_params is not None:
            self.aug_toolkit = AugToolkit(aug_params)

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_path, mask_path = self.sync_img_mask[index]
        output_names = self.output_names[index]

        img = common.gdal_to_numpy(img_path)
        mask = common.gdal_to_numpy(mask_path)

        img = np.append(img, mask, axis=2)

        if self.mode == 'train':
            if self.aug_toolkit is not None:
                img = self.aug_toolkit.run(img)

        mask = img[:, :, -1].copy()
        img = img[:, :, :-1].copy()

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask, output_names

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names


if __name__ == "__main__":
    mean_sar = common.load_mean_file(r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\sar_mean_std\mean_value.txt')
    std_sar = common.load_mean_file(r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\sar_mean_std\std_value.txt')
    mean_opt = common.load_mean_file(r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\opt_mean_std\mean_value.txt')
    std_opt = common.load_mean_file(r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\opt_mean_std\std_value.txt')

    print(mean_sar)
    print(std_sar)
    print(mean_opt)
    print(std_opt)
