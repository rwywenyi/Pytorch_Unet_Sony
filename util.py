import glob
import os
import time

import csv
import rawpy
import numpy as np
import torch
import tqdm
from skimage.metrics import structural_similarity as ssimfunc
from skimage.metrics import peak_signal_noise_ratio as psnrfunc

# 损失函数：平均绝对误差
def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

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

def run_test(model, test_ids, input_dir, gt_dir, epoch, csv_filename):
    psnr = []
    ssim = []
    cnt = 0

    with torch.no_grad():
        model.eval()
        for test_id in test_ids:
            # test the first image in each sequence
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
            for k in range(len(in_files)):
                in_path = in_files[k]
                _, in_fn = os.path.split(in_path)
                print(in_fn)
                gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
                gt_path = gt_files[0]
                _, gt_fn = os.path.split(gt_path)
                in_exposure = float(in_fn[9:-5])
                gt_exposure = float(gt_fn[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)

                raw = rawpy.imread(in_path)
                input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
                # input_full = input_full[:,:512, :512, :]

                im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                # im = im[:1024,:1024]
                scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
                # scale_full = np.minimum(scale_full, 1.0)

                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                # im = im[:1024, :1024]
                gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

                input_full = np.minimum(input_full, 1.0)

                in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).cuda()
                st = time.time()
                cnt +=1
                out_img = model(in_img)
                print('%d\tTime: %.3f'%(cnt, time.time()-st))

                output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                output = np.minimum(np.maximum(output, 0), 1)

                output = output[0, :, :, :]
                gt_full = gt_full[0, :, :, :]
                scale_full = scale_full[0, :, :, :]
                origin_full = scale_full
                scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)  # scale the low-light image to the same mean of the groundtruth
                # import pdb
                # pdb.set_trace()
                # psnr.append(psnrfunc(gt_full[:, :, :], output[:, :, :]))
                # ssim.append(ssimfunc(gt_full[:, :, :], output[:, :, :], channle_axis=-1))

                psnr.append(psnrfunc(gt_full[:, :, :], output[:, :, :]))
                ssim.append(ssimfunc(gt_full[:, :, :], output[:, :, :], multichannel=True))

                print('psnr: ', psnr[-1], 'ssim: ', ssim[-1])

        csv_file = os.path.join(csv_filename, 'results.csv')  # 修改为你想要的CSV文件名

        with open(csv_file, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            print(f'epoch:{epoch}, mean psnr: ', np.mean(psnr))
            print(f'epoch:{epoch}, mean ssim: ', np.mean(ssim))
            csv_writer.writerow([epoch, np.mean(psnr), np.mean(ssim)])
                
        print('done')


        # print(f'epoch:{epoch}, mean psnr: ', np.mean(psnr))
        # print(f'epoch:{epoch}, mean ssim: ', np.mean(ssim))
        # print('done')