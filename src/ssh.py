import mindspore as ms
from mindspore import nn
from mindspore import ops

import numpy as np

from src.vgg16 import vgg16

class ContextModule(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        half_channels = int(out_channels / 2)
        self.conv = [nn.Conv2d(in_channels, half_channels, 3, pad_mode='same', has_bias=True, weight_init='HeUniform'),
                        nn.ReLU()]
        self.channel_1 = [nn.Conv2d(half_channels, half_channels, 3, pad_mode='same', has_bias=True, weight_init='HeUniform'),
                        nn.ReLU()]
        self.channel_2 = [nn.Conv2d(half_channels, half_channels, 3, pad_mode='same', has_bias=True, weight_init='HeUniform'),
                        nn.ReLU()]
        self.channel_2 += [nn.Conv2d(half_channels, half_channels, 3, pad_mode='same', has_bias=True, weight_init='HeUniform'),
                        nn.ReLU()]
        self.conv = nn.SequentialCell(self.conv)
        self.channel_1 = nn.SequentialCell(self.channel_1)
        self.channel_2 = nn.SequentialCell(self.channel_2)
        self.concat = ops.Concat(1)
    
    def construct(self, x):
        x = self.conv(x)
        out_1 = self.channel_1(x)
        out_2 = self.channel_2(x)
        out = self.concat((out_1, out_2))
        return out

        
class DetectModule(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.SequentialCell([nn.Conv2d(in_channels, in_channels, 3, pad_mode='same', has_bias=True, weight_init='HeUniform'),
                        nn.ReLU()])
        self.context_module = ContextModule(in_channels)
        self.cls = nn.Conv2d(2*in_channels, 2, 1, has_bias=True, weight_init='HeUniform')
        self.reg = nn.Conv2d(2*in_channels, 8, 1, has_bias=True, weight_init='HeUniform')
        self.concat = ops.Concat(1)
    
    def construct(self, x):
        conv_out = self.conv(x)
        context_out = self.context_module(x)
        out = self.concat((conv_out, context_out))
        cls_out = self.cls(out)
        reg_out = self.reg(out)
        return cls_out, reg_out


class M1DetectModule(nn.Cell):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, pad_mode='same', has_bias=True, weight_init='HeUniform')
        self.context_model = ContextModule(128, 128)
        self.concat = ops.Concat(1)
        self.cls = nn.Conv2d(2 * in_channels, 2, 1, has_bias=True, weight_init='HeUniform')
        self.reg = nn.Conv2d(2 * in_channels, 8, 1, has_bias=True, weight_init='HeUniform')

    def construct(self, x):
        conv1_out = self.conv1(x)
        context_out = self.context_model(x)
        out = self.concat((conv1_out, context_out))
        cls_out = self.cls(out)
        reg_out = self.reg(out)
        return cls_out, reg_out


class M3DetectModule(nn.Cell):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, 3, pad_mode='same', has_bias=True)
        self.context_model = ContextModule(512, 256)
        self.concat = ops.Concat(1)
        self.cls = nn.Conv2d(2 * in_channels, 2, 1, has_bias=True, weight_init='HeUniform')
        self.reg = nn.Conv2d(2 * in_channels, 8, 1, has_bias=True, weight_init='HeUniform')

    def construct(self, x):
        conv1_out = self.conv1(x)
        context_out = self.context_model(x)
        out = self.concat((conv1_out, context_out))
        cls_out = self.cls(out)
        reg_out = self.reg(out)
        return cls_out, reg_out


class SSH(nn.Cell):
    def __init__(self):
        super().__init__()
        self.m3_max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.m3_detect_model = M3DetectModule()

        self.m2_detect_model = M3DetectModule()

        self.m1_dim_red1 = nn.Conv2d(512, 128, 1, has_bias=True, weight_init='HeUniform')
        self.m1_dim_red2 = nn.Conv2d(512, 128, 1, has_bias=True, weight_init='HeUniform')
        # self.m1_upsampling = nn.ResizeBilinear()
        self.m1_upsampling = ops.ResizeBilinearV2()
        self.m1_conv = nn.SequentialCell([nn.Conv2d(128, 128, 3, pad_mode='same', has_bias=True, weight_init='HeUniform'),
                        nn.ReLU()])
        self.m1_detect_model = M1DetectModule(128)

        self.vgg = vgg16()
    
    def construct(self, x):
        conv4_3, conv5_3 = self.vgg(x)

        m3_max_pooling_out = self.m3_max_pooling(conv5_3)
        m3_out = self.m3_detect_model(m3_max_pooling_out)

        m2_out = self.m2_detect_model(conv5_3)

        m1_dim_red1_out = self.m1_dim_red1(conv4_3)
        m1_dim_red2_out = self.m1_dim_red2(conv5_3)
        # upsampling_out = self.m1_upsampling(m1_dim_red2_out, scale_factor=2)
        upsampling_size = (m1_dim_red1_out.shape[2], m1_dim_red1_out.shape[3])
        upsampling_out = self.m1_upsampling(m1_dim_red2_out, upsampling_size)
        m1_out = upsampling_out + m1_dim_red1_out
        m1_out = self.m1_conv(m1_out)
        m1_out = self.m1_detect_model(m1_out)

        return m1_out, m2_out, m3_out


if __name__ == '__main__':
    img = ms.Tensor(np.zeros([1, 3, 896, 1024]), ms.float32)
    ssh = SSH()
    out = ssh.construct(img)
    print(out[0][0].shape, out[0][1].shape)
    print(out[1][0].shape, out[1][1].shape)
    print(out[2][0].shape, out[2][1].shape)
