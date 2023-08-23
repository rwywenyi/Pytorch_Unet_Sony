import os
import time
import numpy as np
import glob
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import SeeInDark
from EDSR_model import EDSR
from util import reduce_mean, run_test
from dataset_sony import CustomDataset, CustomDatasetMemory

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    opt = {'base_lr': 1e-4, 'batch_size': 1, 'epochs': 4001, 'save_frequency': 100, 'test_frequency': 10000,
           'patch_size': 512, 'load_weigths': False, 'start_epoch':1, 'debug':False}

    print(f"debug mode: {opt['debug']}")
    print(f"batch_size: {opt['batch_size']}")
    print(f"start_epoch: {opt['start_epoch']}")
    print(f"all train epochs: {opt['epochs']}")
    print(f"save_frequency: {opt['save_frequency']}")
    print(f"test_frequency: {opt['test_frequency']}")
    epoch = opt['start_epoch']

    writer = SummaryWriter(log_dir='./logs', filename_suffix='EDSR')

    # 保存模型、日志地址
    metric_average_file = 'metric_test_filename.csv'
    save_file_path = './result'
    file_name = 'result_EDSR_32_SE'
    save_weights_file = os.path.join(save_file_path, file_name, 'weights')
    save_images_file = os.path.join(save_file_path, file_name, 'images')
    save_csv_file = os.path.join(save_file_path, file_name, 'csv_files')
    csv_filename = os.path.join(save_file_path, file_name, 'test_result')

    if not os.path.exists(save_weights_file):
        os.makedirs(save_weights_file)
    if not os.path.exists(save_images_file):
        os.makedirs(save_images_file)
    if not os.path.exists(csv_filename):
        os.makedirs(csv_filename)
    if not os.path.exists(save_csv_file):
        os.makedirs(save_csv_file)

    # TODO 训练数据地址, 尽量使用相对路径
    # TODO glob.glob的结果是list！！！
    input_dir = './Sony/Sony/short/'
    gt_dir = './Sony/Sony/long/'
    train_gt_paths = glob.glob(gt_dir + '0*.ARW')
    train_ids = []
    for i in range(len(train_gt_paths)):
        if opt['debug']:
            if i > 12:
                break
        _, train_fn = os.path.split(train_gt_paths[i])
        train_ids.append(int(train_fn[0:5]))

    test_path = glob.glob(input_dir + '1*_00_*.ARW')
    test_ids = []
    for i in range(len(test_path)):
        if opt['debug']:
            if i > 12:
                break
        _, test_fn = os.path.split(test_path[i])
        test_ids.append(int(test_fn[0:5]))

    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images=[None]*6000
    input_images = {}
    input_images['300'] = [None]*len(train_ids)
    input_images['250'] = [None]*len(train_ids)
    input_images['100'] = [None]*len(train_ids)

    # 指定训练设备,构建dataloader
    # train_dataset = CustomDataset(train_ids, input_dir, gt_dir, patch_size=opt['patch_size'])
    train_dataset = CustomDatasetMemory(train_ids, input_dir, gt_dir, input_images, gt_images, patch_size=opt['patch_size'])
    # test_dataset = CustomDataset(test_ids, input_dir, gt_dir, training=False)
    dataloader_train = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=1,pin_memory=True)
    # dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    import argparse
    # Argument for EDSR
    parser = argparse.ArgumentParser(description='EDSR')
    parser.add_argument('--n_resblocks', type=int, default=32,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--scale', type=str, default=2,
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='output patch size')
    parser.add_argument('--n_colors', type=int, default=4,
                        help='number of input color channels to use')
    parser.add_argument('--o_colors', type=int, default=3,
                        help='number of output color channels to use')
    args = parser.parse_args()

    model = SeeInDark()
    # model = EDSR(args)

    if opt['load_weigths']:
        model_weights = torch.load('/hy-tmp/learning_to_see_in_the_dark/saved_model/checkpoint_sony_e0500.pth')
        model.load_state_dict(model_weights)

    # 如果是SeeInDark,则使用初始化权重
    model.initialize_weights()
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt['base_lr'])
    optimizer.zero_grad()

    loss_list = ['loss_rgb']
    metrics = ['PSNR_rgb, SSIM_rgb']
    epoch_list = ['Iteration']

    epoch_LR = ['Iter_LR']

    print(f'model: {model.name}')
    print(f"Device: {device}")
    print(f'训练图像： {len(train_ids)}张')
    print(f'测试图像： {len(test_ids)}张')

    while epoch < opt['epochs']:
        print(f'epoch:{epoch:04}...........................')
        if epoch > 2000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
                print('降低学习率..............')
        st = time.time()
        for x, img in tqdm.tqdm(enumerate(dataloader_train)):
            input_raw = img[0].to(device)
            gt_rgb = img[1].to(device)
            ratio = img[2].cpu().data.numpy()
            train_id = img[3].cpu().data.numpy()

            # import pdb
            # pdb.set_trace()

            model.train()
            pred_rgb = model(input_raw)

            loss = reduce_mean(pred_rgb, gt_rgb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train_loss', loss.item(), epoch)

            # 调整维度顺序，BCHW -》 BHWC，放cpu上并且转为numpy
            pred_rgb = pred_rgb.permute(0, 2, 3, 1).cpu().data.numpy()
            gt_rgb = gt_rgb.permute(0, 2, 3, 1).cpu().data.numpy()

            pred_rgb_four = pred_rgb
            gt_rgb_four = gt_rgb
            BN = gt_rgb_four.shape[0]

            # 四维转三维，成正常的RGB三通道
            pred_rgb = pred_rgb[0, :, :, :]
            gt_rgb = gt_rgb[0, :, :, :]

            if epoch % opt['save_frequency'] == 0:
                # TODO 带有batch size的四维变量计算ssim和psnr与分离后分别计算求平均有差异
                psnr_rgb = PSNR(gt_rgb_four, np.clip(pred_rgb_four, 0, 1))  # 9.94866529903103
                ssim_rgb = SSIM(gt_rgb * 255, np.clip(pred_rgb * 255, 0, 255), data_range=255, channel_axis=-1)

                print('\nepoch:%d, loss_rgb:%.4f, PSNR_rgb:%.4f, SSIM_rgb:%.4f' % (
                    epoch, np.mean(loss.cpu().data.numpy()), psnr_rgb, ssim_rgb))

                loss_list.append('{:.5f}'.format(loss.item()))
                metrics.append('{:.5f},{:.5f}'.format(np.mean(psnr_rgb), np.mean(ssim_rgb)))
                epoch_list.append(epoch)
                epoch_LR.append(optimizer.param_groups[0]['lr'])
                np.savetxt(os.path.join(save_csv_file, 'train_log.csv'),
                           [p for p in zip(epoch_list, epoch_LR, loss_list, metrics)], delimiter=',', fmt='%s')

                epoch_result_dir = os.path.join(save_images_file, f'{epoch:04}/')
                if not os.path.isdir(epoch_result_dir):
                    os.makedirs(epoch_result_dir)

                for i in range(BN):
                    # TODO 对模型生成的RGB图像的上下边界进行压缩！！！
                    pred_rgb_four[i, :, :, :] = np.minimum(np.maximum(pred_rgb_four[i, :, :, :], 0), 1)
                    temp = np.concatenate((gt_rgb_four[i, :, :, :], pred_rgb_four[i, :, :, :]), axis=1)
                    Image.fromarray((temp * 255).astype('uint8')).save(epoch_result_dir + f'{train_id[i]:05}_00_{ratio[i]}.jpg')

        if epoch % opt['save_frequency'] == 0:
            torch.save({'model': model.state_dict()},
                       os.path.join(save_weights_file, 'weights_{}.pth'.format(epoch)))
            print('model saved......')
        print(f'epoch:{epoch:04}耗时:{time.time() - st:.3}s')
        
        if epoch % opt['test_frequency'] == 0:
            test_start_time = time.time()
            # run_test(model, dataloader_test, save_test_file, metric_average_file, device)
            run_test(model, test_ids, input_dir, gt_dir, epoch, csv_filename)
            print(f'test耗时: {time.time() - test_start_time:.3}')
        
        epoch += 1
