import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from misc.utils import *
from models.Mamba.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class ScaleAwareGate(nn.Module):
    def __init__(self, inp, oup):
        super(ScaleAwareGate, self).__init__()

        self.local_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(oup)

        self.global_embedding = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(oup)

        self.global_act = nn.Conv2d(inp, oup, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(oup)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        local_feat = self.bn1(local_feat)

        global_feat = self.global_embedding(x_g)
        global_feat = self.bn2(global_feat)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        global_act = self.global_act(x_g)
        global_act = self.bn3(global_act)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out

class FPN(nn.Module):
    def __init__(
            self,
            # Initial Basic dims ===========
            dim_m=896,
            dim=512,
            channels=[128, 256, 512],
            # basic dims ===========
            d_model=512,
            d_state=1,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=False,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v05_noz",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        super(FPN, self).__init__()
        self.dim_m = dim_m
        self.dim = dim
        self.channels = channels
        self.down_channel = nn.Conv2d(self.dim_m, self.dim, 1)
        self.up_channel = nn.Conv2d(self.dim, self.dim_m, 1)
        self.ss2d = SS2D(d_model, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min,
                         dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type, channel_first, **kwargs)
        self.fusion = nn.ModuleList([
            ScaleAwareGate(channels[i], channels[i])
            for i in range(len(channels))
        ])

    def forward(self, inputs):
        prev_shape = inputs[2].shape[2:]
        in_0 = F.interpolate(inputs[0], size=prev_shape, mode='nearest')
        in_1 = F.interpolate(inputs[1], size=prev_shape, mode='nearest')
        in_2 = inputs[2]
        input = torch.cat([in_0,in_1,in_2], dim=1)
        input = self.down_channel(input)
        input = input.permute(0,2,3,1)
        output = self.ss2d(input)
        # output = input
        output = output.permute(0,3,1,2)
        output = self.up_channel(output)
        xx = output.split(self.channels, dim=1)
        results = []
        for i in range(len(self.channels)):
            Mix_before = inputs[i]
            Mix_after = xx[i]
            out_ = self.fusion[i](Mix_before, Mix_after)
            results.append(out_)
        return results

if __name__ == '__main__':
    model = FPN()
    x1 = torch.randn(2, 128, 128, 128)
    x2 = torch.randn(2, 256, 64, 64)
    x3 = torch.randn(2, 512, 32, 32)
    x = [x1, x2, x3]
    y = model(x)
    print("y:",y[0].shape,y[1].shape,y[2].shape)