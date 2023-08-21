import numpy as np
import rawpy
import glob

import torch
from torch.utils.data import Dataset


# pack Bayer image to 4 channels
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def image_read(short_expo_files, long_expo_files):
    # 根据short和long的对应关系计算亮度增益
    if short_expo_files[-6] == '3':
        ratio = 300
    elif short_expo_files[-6] == '4':
        ratio = 250
    elif long_expo_files[-7] == '3':
        ratio = 300
    else:
        ratio = 100

    raw = rawpy.imread(short_expo_files)
    short_img = pack_raw(raw) * ratio

    gt_raw = rawpy.imread(long_expo_files)
    long_img = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    long_img = np.float32(long_img / 65535.0)
    return short_img, long_img, ratio


class CustomDataset(Dataset):
    def __init__(self, train_ids, input_dir, gt_dir, patch_size=512, training=True):
        self.train_ids = train_ids
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = patch_size
        self.training = training

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        if self.training:
            train_id = self.train_ids[idx]
            train_input_dir = glob.glob(self.input_dir + f'{train_id:05}_00*ARW')
            train_input_dir = train_input_dir[np.random.randint(0, len(train_input_dir))]
            train_gt_dir = glob.glob(self.gt_dir + '%05d_00*.ARW' % train_id)[0]
            short_img, long_img, ratio = image_read(train_input_dir, train_gt_dir)

            # crop
            h = short_img.shape[0]
            w = short_img.shape[1]
            xx = np.random.randint(0, w - self.ps)
            yy = np.random.randint(0, h - self.ps)
            input_patch = short_img[yy:yy + self.ps, xx:xx + self.ps, :]
            gt_patch = long_img[yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip 左右镜像
                input_patch = np.flip(input_patch, axis=0)
                gt_patch = np.flip(gt_patch, axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose 旋转90度
                input_patch = np.transpose(input_patch, (1, 0, 2))
                gt_patch = np.transpose(gt_patch, (1, 0, 2))

            # 限制patch的范围在0~1
            input_patch = np.minimum(input_patch, 1.0)
            gt_patch = np.maximum(gt_patch, 0.0)
        else:
            test_ids = self.train_ids[idx]
            test_id = test_ids[0:5]
            test_input_dir = glob.glob(self.input_dir + test_ids)[0]
            test_gt_dir = glob.glob(self.gt_dir + f'{test_id}_00*.ARW')[0]
            short_img, long_img, ratio = image_read(test_input_dir, test_gt_dir)

            input_patch = short_img
            gt_patch = long_img

        in_img = torch.from_numpy(input_patch).permute(2, 0, 1)  # H W C -> C H W
        gt_img = torch.from_numpy(gt_patch).permute(2, 0, 1)
        return in_img, gt_img, ratio