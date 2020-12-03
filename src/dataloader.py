import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from utils.utils import irregular_mask


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist,
                 irr_mask_path=None, seg_mask_path=None,
                 fix_mask_path=None, training=True):
        super(InpaintingDataset, self).__init__()
        self.config = config
        self.training = training
        self.data = self.load_flist(flist)
        self.data = sorted(self.data, key=lambda x: x.split('/')[-1])
        self.irr_masks = []
        self.seg_masks = []
        self.fix_masks = []
        if irr_mask_path is not None:
            self.irr_masks = self.load_flist(irr_mask_path)
        if seg_mask_path is not None:
            self.seg_masks = self.load_flist(seg_mask_path)
        if fix_mask_path is not None:
            self.fix_masks = self.load_flist(fix_mask_path)
        if not training:
            assert len(self.fix_masks) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.load_item(index)

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.config.input_size
        # load image
        img = cv2.imread(self.data[index])[:, :, ::-1]

        # resize/crop if needed
        img = self.resize(img, size, size, center_crop=self.config.center_crop)

        # load mask
        mask = self.load_mask(img, index)

        # augment data
        if self.training is True and self.config.flip is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...]
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...]
            if random.random() < 0.5:
                mask = mask[::-1, :, ...]

        img = self.to_tensor(img, norm=True)  # norm to -1~1
        mask = self.to_tensor(mask)

        meta = {'img': img, 'mask': mask}

        return meta

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.fix_masks[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 50% mask with random brush, 50% mask with
            if random.random() < self.config.hyper_mask_rate:
                mask_index = random.randint(0, len(self.irr_masks) - 1)
                mask = cv2.imread(self.irr_masks[mask_index], cv2.IMREAD_GRAYSCALE)
            else:
                mask_index = random.randint(0, len(self.seg_masks) - 1)
                mask = cv2.imread(self.seg_masks[mask_index], cv2.IMREAD_GRAYSCALE)
            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, (height, width))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort(key=lambda x: x.split('/')[-1])
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
