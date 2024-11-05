import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math
import random
from model.ops.modules import MSDeformAttn
from model.backbone.positional_encoding import SinePositionalEncoding
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)


class MyCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., q_ratio=1,
                 kv_ratio=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim*2, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj_0 = nn.Linear(dim, dim)
        self.proj_drop_0 = nn.Dropout(proj_drop)

        self.q_ratio = q_ratio
        self.kv_ratio = kv_ratio
        if self.q_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=q_ratio, stride=q_ratio)
            self.norm = nn.LayerNorm(dim)
        if self.kv_ratio > 1:
            self.sr_kv = nn.Conv2d(dim, dim, kernel_size=kv_ratio, stride=kv_ratio)
            self.norm_kv = nn.LayerNorm(dim)

        self.apply(self._init_weights)

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

    def forward(self, q, kv, H, W):
        B, N, C = q.shape  # 4,4096,256  #4,20480,256  #s_x 4,5,4096,256  prior_mask=4,4096,1是相似度余弦相似度
        # prior_mask = prior_mask.expand_as(q)
        if self.kv_ratio > 1: #4
            new_kv = self.sr_kv(kv.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, -1).permute(0, 2, 1)
            new_kv = self.norm_kv(new_kv)
            # _,new_N,_ = new_query.shape
            kv = self.kv(new_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 4,8,256,32
            k, v = kv[0], kv[1]  # B,8,256,32
        else:
            kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 4,8,256,32
            k, v = kv[0], kv[1]  # B,8,256,32

        if self.q_ratio > 1:
            x_ = q.permute(0, 2, 1).reshape(B, C, H, W)  # 4,256,64,64
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # 4,256,256  #downsample
            x_ = self.norm(x_)  # 4,256,256
            q = self.q(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 8,8,4096,32 #8,8,256,32
        else:
            q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        _, _, new_N, _ = q.shape
        attn_ = (q @ k.transpose(-2, -1))  # 8,8,4096,32---32,K_N=256  8,8,4096,256
        attn = (attn_ * self.scale).softmax(dim=-1)  # 8,8,4096,256  #,K_N=256维度加起来=1   support的数量是256
        # 4,8,4096,256  #4096维度加起来=1
        attn = self.attn_drop(attn)
        q2k_sim_idx= attn_.max(3)[1]  # 8,8,4096
        q2k_sim_idx=q2k_sim_idx.unsqueeze(3).expand_as(q)  #8,8,4096，32  #np.array(q2k_sim_idx.data.cpu())
        re_q = torch.gather(k, 2, q2k_sim_idx)#4，8，4096，32
        re_q = re_q.transpose(1, 2).reshape(B, new_N, C)#B,256,256

        x = (attn @ v).transpose(1, 2).reshape(B, new_N, C)  # 4,256,256
        x = self.proj_0(x)
        x = self.proj_drop_0(x)
        x = x + q.transpose(1, 2).reshape(B, new_N, C)
        re_q = re_q + q.transpose(1, 2).reshape(B, new_N, C)
        x = torch.cat([x, re_q], 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.q_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H // self.q_ratio, W // self.q_ratio)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True).reshape(B, C, -1).permute(0, 2, 1)

        return x


class SelfTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384,
                 num_heads=8,
                 num_layers=1,
                 num_levels=1,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 ):
        super(SelfTransformer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.num_points = num_points
        self.use_ffn = use_ffn
        self.feedforward_channels = embed_dims * 3
        self.dropout = dropout

        self.qry_self_layers = []
        self.layer_norms = []
        self.ffns = []
        for l_id in range(self.num_layers):
            self.qry_self_layers.append(
                MSDeformAttn(embed_dims, num_levels, 12 if embed_dims % 12 == 0 else self.num_heads, num_points)
            )
            self.layer_norms.append(nn.LayerNorm(embed_dims))

            if self.use_ffn:
                self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                self.layer_norms.append(nn.LayerNorm(embed_dims))

        self.qry_self_layers = nn.ModuleList(self.qry_self_layers)
        if self.use_ffn:
            self.ffns = nn.ModuleList(self.ffns)
        self.layer_norms = nn.ModuleList(self.layer_norms)

        self.positional_encoding = SinePositionalEncoding(embed_dims // 2, normalize=True)
        self.level_embed = nn.Parameter(torch.rand(num_levels, embed_dims))
        nn.init.xavier_uniform_(self.level_embed)

        self.proj_drop = nn.Dropout(dropout)

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = []
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []
        for lvl in range(self.num_levels):
            src = x[lvl]
            bs, c, h, w = src.shape

            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id] == 255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w))

            pos_embed = self.positional_encoding(qry_valid_mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1)  # [bs, num_elem, c]
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1)  # [bs, num_elem]
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1)  # [bs, num_elem, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [num_lvl, 2]
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [num_lvl]

        return src_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index

    def forward(self, x, qry_masks):
        if not isinstance(x, list):  # 4,256,64,64   #prior_mask, 8,1,,64,64
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]
        assert len(x) == len(qry_masks) == self.num_levels
        bs, c, h, w = x[0].size()

        x_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(
            x, qry_masks)
        ##4,5,4096,256;  4,20480,256;（20480=4096*5）  4,20480;  4,20480
        reference_points = self.get_reference_points(spatial_shapes, device=x_flatten.device)
        # 1,4096,1,2
        q = x_flatten
        pos = pos_embed_flatten

        ln_id = 0
        ffn_id = 0

        for l_id in range(self.num_layers):  # 2
            q = q + self.proj_drop(
                self.qry_self_layers[l_id](q + pos, reference_points, q, spatial_shapes, level_start_index,
                                           qry_valid_masks_flatten))
            q = self.layer_norms[ln_id](q)  # 4，4096，256
            ln_id += 1

            if self.use_ffn:
                q = self.ffns[ffn_id](q)
                ffn_id += 1
                q = self.layer_norms[ln_id](q)
                ln_id += 1
        qry_feat = q.permute(0, 2, 1).view(bs, c, h, w)

        return qry_feat

