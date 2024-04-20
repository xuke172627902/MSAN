'''
    this code is based on basicsr 1.4.2 and VapSR(2022 ECCVW)
'''
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

import torch


class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()

        self.conv_7 = nn.Sequential(nn.Conv2d(dim, dim, (5, 1), 1, (2, 0), groups=dim),
                                    nn.Conv2d(dim, dim, (1, 5), 1, (0, 2), groups=dim))
        self.conv_11 = nn.Sequential(nn.Conv2d(dim, dim, (11, 1), 1, (5, 0), groups=dim),
                                     nn.Conv2d(dim, dim, (1, 11), 1, (0, 5), groups=dim))
        self.conv_21 = nn.Sequential(nn.Conv2d(dim, dim, (15, 1), 1, (7, 0), groups=dim),
                                     nn.Conv2d(dim, dim, (1, 15), 1, (0, 7), groups=dim))

        self.mixer = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        c7 = self.conv_7(x)
        c11 = self.conv_11(x)
        c21 = self.conv_21(x)

        add = x + c7 + c11 + c21
        output = self.mixer(add)

        return output


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.msca = MSCA(dim=dim)

    def forward(self, x):
        u = x.clone()
        attn = self.msca(x)

        return u * attn

class MBAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.conv = nn.Sequential(nn.Conv2d(d_atten, d_atten, 1, 1),
                                  nn.Conv2d(d_atten, d_atten, 3, 1, 1, groups=d_atten))
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x

def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

#both scale X2 and X3 use this version
def pixelshuffle_single(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])


def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)

@ARCH_REGISTRY.register()
class MSAN(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, d_atten=64, conv_groups=1):
        super(MSAN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(MBAB, num_block, num_feat, d_atten)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=conv_groups)

        # upsample
        if scale == 4:
            self.upsampler = pixelshuffle(num_feat, num_out_ch, upscale_factor=scale)
        else:
            self.upsampler = pixelshuffle_single(num_feat, num_out_ch, upscale_factor=scale)

    def forward(self, feat):
        feat = self.conv_first(feat)
        body_feat = self.body(feat)
        body_out = self.conv_body(body_feat)
        feat = feat + body_out
        out = self.upsampler(feat)
        return out


if __name__ == '__main__':
    from nni.compression.pytorch.utils.counter import count_flops_params
    x = torch.rand((1, 3, 426, 240))
    model = MSAN(num_in_ch=3, num_out_ch=3, scale=3, num_feat=48, num_block=14, d_atten=64, conv_groups=1)

    flops, params, results = count_flops_params(model, x)
    x = model(x)
    print(x.shape)