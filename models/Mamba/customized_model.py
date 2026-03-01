import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
#from timm.models.layers import DropPath
from timm.layers import DropPath
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import einops

from .vmamba import SS2D, VSSM, LayerNorm2d, Linear2d


class Twister(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
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
            forward_type="v2",
            channel_first=False,
            # ======================
            input_res=64,
            **kwargs,
    ):
        super().__init__()
        self.input_res = input_res
        self.ss2d = SS2D(d_model, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min,
                         dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type, channel_first, **kwargs)
        self.ss2d.in_proj = Linear2d(d_model * 3, self.ss2d.in_proj.weight.shape[0], bias=bias)
        # self.input_res = min(input_res, 16)
        forward_type_1d = "v052d"
        self.ss1d = SS2D(self.input_res ** 2, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias,
                         dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type_1d, channel_first,
                         **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        x_prepaired = x_mix = x.view(B, -1, H*W)
        # x_mix = F.interpolate(x, size=(self.input_res,self.input_res), mode='bilinear') 
        x_mix = x_mix.permute(0, 2, 1).unsqueeze(-2).contiguous()  # b, hw, 1, c
        x_mix = self.ss1d(x_mix)
        x_mix = x_mix.squeeze(-2).permute(0, 2, 1).contiguous()  # b, c, hw
        # x_mix = F.interpolate(x_mix.view(B,-1,self.input_res,self.input_res), size=(H,W), mode='bilinear')

        x = x_mix + x_prepaired

        out = self.ss2d(x)

        return out


class SCBlock(nn.Module):
    def __init__(
            self,
            forward_coremm='SS2D',
            dim=256,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        norm_layer = norm_layer
        dim = dim
        drop_path = drop_path
        self.ln_1 = norm_layer(dim)
        self.forward_coremm = forward_coremm

        self.self_attention = Twister(
            d_model=dim,
            d_state=1,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=False,
            dropout=0.0,
            initialize="v0",
            forward_type="v05_noz",
            channel_first=True,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input = input.permute(0,2,3,1)
        out = self.ln_1(input)
        out = self.self_attention(out)
        out = input + self.drop_path(out)
        x = out.permute(0,3,1,2)
        return x

if __name__ == "__main__":
    input = torch.randn(4,32,32,256).cuda()
    model = SCBlock().cuda()
    output = model(input)
    print("output:",output.shape)