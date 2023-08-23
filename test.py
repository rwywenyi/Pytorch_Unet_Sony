import numpy as np
import torch
from torch.nn.functional import l1_loss
from skimage.metrics import structural_similarity
import pdb


import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import SeeInDark, YourNetwork
fake_img = torch.randn(1, 4, 512, 512)
# output = torch.nn.functional.pixel_shuffle(input, 3)
# print(output.size())


writer = SummaryWriter(log_dir='./logs', filename_suffix='unet')
# writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

# 模型
# fake_img = torch.randn(1, 3, 32, 32)  # 生成假的图片作为输入

lenet = SeeInDark()  # 以LeNet模型为例

writer.add_graph(lenet, fake_img)  # 模型及模型输入数据

writer.close()
