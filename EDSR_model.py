import torch
from tensorboard import summary

import common
import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.name = 'EDSR'
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

if __name__ == '__main__':
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

    model = EDSR(args).to('cpu')
    print(model)
    input = torch.rand((1, 4, 512, 512)).to('cpu')
    print(model(input).shape)
    # summary(model, (4, 512, 512), device="cpu")
    # input = torch.ones(2,2,3,4)
    # trans = nn.ConvTranspose2d(2, 4, 3, stride=2, padding=2)
    # print(trans(input).shape)