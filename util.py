import os

import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

# 损失函数：平均绝对误差
def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


def run_test(model, dataloader_test, save_csv_file, metric_test_filename, device):
    psnr_list = ['PSNR_1']
    ssim_list = ['SSIM_1']
    loss_list = ['loss']
    count_list = ['count']
    count = 0
    with torch.no_grad():
        model.eval()
        for image_num, img in tqdm.tqdm(enumerate(dataloader_test)):
            input_raw = img[0].to(device)
            gt_rgb = img[1].to(device)
            ratio = img[2]
            pred_rgb = model(input_raw)
            loss = reduce_mean(pred_rgb, gt_rgb)

            # 调整维度顺序，BCHW -》 BHWC，放cpu上并且转为numpy
            pred_rgb = pred_rgb.permute(0,2,3,1).cpu().data.numpy()
            gt_rgb = gt_rgb.permute(0,2,3,1).cpu().data.numpy()
            # 四维转三维，成正常的RGB三通道
            pred_rgb = pred_rgb[0,:,:,:]
            gt_rgb = gt_rgb[0,:,:,:]

            psnr_rgb_img = PSNR(gt_rgb, np.clip(pred_rgb, 0, 1))
            ssim_rgb_img = SSIM(gt_rgb * 255, np.clip(pred_rgb * 255, 0, 255), data_range=255, channel_axis=-1)

            print(f'loss: {loss.item():.5f}')
            print(f'psnr: {psnr_rgb_img}')
            print(f'ssim: {ssim_rgb_img}')

            loss_list.append('{:.5f}'.format(loss.item()))
            psnr_list.append(psnr_rgb_img)
            ssim_list.append(ssim_rgb_img)
            count += 1
            count_list.append(count)
            np.savetxt(os.path.join(save_csv_file, metric_test_filename), [p for p in zip(count_list, psnr_list, ssim_list)],
                       delimiter=',', fmt='%s')