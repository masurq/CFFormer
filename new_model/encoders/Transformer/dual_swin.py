import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from collections import OrderedDict
from thop import clever_format, profile

from models.new_model.modules import FeatureFusion as FFM
from models.new_model.modules import FeatureCorrection_s2c as FCM


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将feature_map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: B H//M M W//H M C -> B H//M W//M M M C
    # view: B H//M W//M M M C -> B*num_windows Mh Mw C
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将一个个window还原成一个feature_map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: B*num_windows Mh Mw C -> B H//M W//M M M C
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: B H//M W//M M M C -> B H//M M W//M M C
    # view: B H//M M W//M M C -> B H W C
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # 定义一个可训练的相对位置偏置参数，零矩阵初始化
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 2 Mh*Mw 1 -> 2 1 Mh*Mw，相当于unsqueeze新插入一个维度
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 固定参数值存入缓存中
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # Batch_size*num_windows, Mh*Mw, total_embed_dim
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 通过广播机制相加
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 确保feature能够被window_size整除, 只在右下侧padding
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        # torch.roll滚动操作，例如[1, 2, 3, 4, 5]经过torch.roll(x, shifts=2, dims=0)变为[4, 5, 1, 2, 3]
        # 负数代表从左往右和从上往下
        # 不改变张量的形状和大小，只是对元素重新排列。滚动操作也不会修改原始张量，而是返回一个新的滚动后的张量
        # 如果shift_size>0,则执行SW-MSA,否则执行W-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # 将同质区域赋值后划分为一个个windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_windows*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nWs*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nWs*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # nWs*B, window_s, window_s, C
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        # 还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面的pad移除
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2  # shift_window偏移量
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 判断是否使用SW-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍，先向上取整再×原值
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature_map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        # 以4*4的window为例，这样shift_size为4/2=2
        # shift_window后的切片，在h维度先是0到-4，然后再-4到-2，再-2到0，分成了三块；w维度也是一样
        # 这种切分是为了给同质区域赋相同的值，所谓不同质是指shift后不连续的图像块(在shift前是连续的)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        # 循环刚刚获取到的同质区域的位置并对其进行赋值，比如h[0,-4]到w[0,-4]赋值为0, h[0,-4]到w[-4,-2]赋值为1......
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nWs, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # nWs Mh*Mw
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 通过该步骤获取同质区域为0, nWs 1 Mh*Mw - nWs Mh*Mw 1
        # 最终获取的到了mask蒙版, 同质区域赋值0, 非同质区域赋值-100, 这样经过softmax则变为0, nWs Mh*Mw Mh*Mw
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        return x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        # (W_left, W_right, H_top, H_bottom, C_front, C_back)
        if W % self.patch_size[1] != 0:  # 宽度方向右侧padding
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:  # 高度方向底部padding
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)  # B C N(HW)->B N C
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Wh, Ww)  # B C H W

        return x


class DualSwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,  # Block中间层的channel为输入channel的mlp_ratio倍
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 norm_fuse=nn.BatchNorm2d,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 sr_ratios=[8, 4, 2, 1],
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        # 包括Patch Partition和Linear Embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # split depth into non-overlapping patches
        self.patch_embed_d = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

            self.absolute_pos_embed_d = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed_d, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers and fusemodules
        self.layers = nn.ModuleList()
        self.layers_d = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.downsamples_d = nn.ModuleList()
        self.FCMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里不包含PatchMerging,也就是downsample,其他的和原swin一样
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

            layer_d = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            self.layers_d.append(layer_d)

            fr = FCM(dim=int(embed_dim * 2 ** i_layer), reduction=1)
            self.FCMs.append(fr)
            # patch merging layer
            # 前三个Block含有downsample，最后一个没有
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(dim=int(embed_dim * 2 ** i_layer), norm_layer=norm_layer)
                self.downsamples.append(downsample)
                downsample_d = PatchMerging(dim=int(embed_dim * 2 ** i_layer), norm_layer=norm_layer)
                self.downsamples_d.append(downsample_d)

            fuse = FFM(dim=int(embed_dim * 2 ** i_layer), reduction=1, num_heads=num_heads[i_layer],
                       norm_layer=norm_fuse, sr_ratio=sr_ratios[i_layer])
            self.FFMs.append(fuse)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

            layer_d = norm_layer(num_features[i_layer])
            layer_name_d = f'norm_d{i_layer}'
            self.add_module(layer_name_d, layer_d)

        self.apply(self._init_weights)
        self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, x_d):
        """Forward function."""
        # B C H W
        x = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)

        # Wh和Ww是原图经过4倍下采样后的feature的高宽
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            absolute_pos_embed_d = F.interpolate(self.absolute_pos_embed_d, size=(Wh, Ww), mode='bicubic')
            x_d = (x_d + absolute_pos_embed_d).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)
        #  B N C
        x = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer_d = self.layers_d[i]

            x, H, W = layer(x, Wh, Ww)
            x_d, _, _ = layer_d(x_d, Wh, Ww)

            # Feature Rectify
            # B HW C -> B C H W
            x = x.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            x_d = x_d.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            x, x_d = self.FCMs[i](x, x_d)
            # B C H W ->  B HW C
            x = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)

            x_out, x_out_d = x, x_d

            if i < self.num_layers - 1:
                x = self.downsamples[i](x, H, W)
                x_d = self.downsamples_d[i](x_d, H, W)
                # 如果是奇数则加一除二为整数，偶数加一向下取整也为整数
                # 现在的Wh和Ww为经过两倍下采样后的feature的高宽
                Wh, Ww = (H + 1) // 2, (W + 1) // 2

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                norm_layer_d = getattr(self, f'norm_d{i}')
                x_out_d = norm_layer_d(x_out_d)

                # B HW C -> B C H W
                x_out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                x_out_d = x_out_d.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                out = self.FFMs[i](x_out, x_out_d)

                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DualSwinTransformer, self).train(mode)
        self._freeze_stages()


def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        # raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('downsample') >= 0 and k.find('layer') >= 0:
            name = k.replace('downsample.', '')
            name = name.replace('layers', 'downsamples')
            state_dict[name] = v
            name = name.replace('downsamples', 'downsamples_d')
            state_dict[name] = v
        elif k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'patch_embed_d')] = v
        elif k.find('layer') >= 0:
            state_dict[k] = v
            state_dict[k.replace('layers', 'layers_d')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'norm_d')] = v

    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    del state_dict
    t_end = time.time()
    print("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))


class swin_t(DualSwinTransformer):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    def __init__(self, **kwargs):
        super(swin_t, self).__init__(pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                                     depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                     attn_drop_rate=0.3, drop_path_rate=0.1,
                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                                     frozen_stages=-1, use_checkpoint=False, **kwargs)


class swin_s(DualSwinTransformer):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    def __init__(self, **kwargs):
        super(swin_s, self).__init__(pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                                     depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                     attn_drop_rate=0.3, drop_path_rate=0.1,
                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                                     frozen_stages=-1, use_checkpoint=False, **kwargs)


class swin_b(DualSwinTransformer):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    def __init__(self, **kwargs):
        super(swin_b, self).__init__(pretrain_img_size=384, patch_size=4, in_chans=3, embed_dim=128,
                                     depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                     window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                     attn_drop_rate=0.3, drop_path_rate=0.1,
                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                                     frozen_stages=-1, use_checkpoint=False, **kwargs)


class swin_l(DualSwinTransformer):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    def __init__(self, **kwargs):
        super(swin_l, self).__init__(pretrain_img_size=384, patch_size=4, in_chans=3, embed_dim=192,
                                     depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                                     window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                     attn_drop_rate=0.3, drop_path_rate=0.1,
                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                                     frozen_stages=-1, use_checkpoint=False, **kwargs)


if __name__ == '__main__':
    model = swin_t().cuda()
    # print(model)
    left = torch.randn(1, 3, 256, 256).cuda()
    right = torch.randn(1, 3, 256, 256).cuda()

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
# 1K数据集只训练了tiny、small和base,22K只训练了base和large
# 其中只有base和large有window12_384版本，window越大，序列长度越小，所消耗的资源也越小
# def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=12,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
#     # trained ImageNet-22K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
#     # trained ImageNet-22K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=192,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(6, 12, 24, 48),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
