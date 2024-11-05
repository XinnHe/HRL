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

class MyCrossAttention_ori(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.ass_drop   = nn.Dropout(0.1)

        self.drop_prob = 0.1


    def forward(self, q, k, v, supp_valid_mask=None, supp_mask=None, cyc=True):
        B, N, C = q.shape #N=4096;B 4;C 256
        N_s = k.size(1) #2738  #q=query kv=support
        # #k 4,2738,356     sampled_valid_mask:4,2738  sampled_mask:4,2738;
        q = self.q_fc(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #4,8,4096,32
        #4,8,4096,32
        k = self.k_fc(k).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#4,8,2738,32
        v = supp_mask.unsqueeze(1).unsqueeze(-1).float().repeat(1,self.num_heads, 1,C // self.num_heads) #4,8,2738,32 #1,32,2738,1
        #v = self.v_fc(v).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#4,8,2738,32.transpose(0, 1).contiguous()

        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(1).repeat(1, self.num_heads, 1) #4,8,2738 [bs, nH, n]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [bs, nH, n, n]#4,8,4096,2738
        '''
        if supp_mask is not None and cyc==True:
            association = []
            for hd_id in range(self.num_heads):
                attn_single_hd = attn[:, hd_id, ...]#4,4096,2738
                k2q_sim_idx = attn_single_hd.max(1)[1] # [bs, n] #返回最大值的索引#4,2738  i*

                q2k_sim_idx = attn_single_hd.max(2)[1] # [bs, n] #4,4096  j*

                re_map_idx = torch.gather(q2k_sim_idx, 1, k2q_sim_idx) # k2q_sim_idx作为索引找 4，2738
                re_map_mask = torch.gather(supp_mask, 1, re_map_idx)

                asso_single_head = (supp_mask == re_map_mask).to(attn.device) #4，2738 [bs, n], True means matched position in supp
                association.append(asso_single_head.unsqueeze(1))#4，1，2738
            association = torch.cat(association, dim=1) # [bs, nH, ns] 4，8，2738

        if cyc:
            inconsistent = ~association #取反
            inconsistent = inconsistent.float()#变0. 1.
            inconsistent = self.ass_drop(inconsistent)
            supp_valid_mask[inconsistent>0] = 1. #把不一致的地方设为1
            

        if supp_valid_mask is not None:
            supp_valid_mask = supp_valid_mask.unsqueeze(-2).float() # [bs, nH, 1, ns]
            supp_valid_mask = supp_valid_mask * -10000.0 #把不一致的地方变无穷小
            attn = attn + supp_valid_mask
        '''
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) ##4,8,4096,2738

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MyCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        '''
        self.merge_conv3 = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim*2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim*2, self.dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )'''
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            self.sr_kv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
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

    def forward(self, query,prior_mask,s_x, H, W):
        B, N, C = query.shape #4,4096,256  #4,20480,256  #s_x 4,5,4096,256  prior_mask=4,4096,1是相似度余弦相似度
        #prior_mask = prior_mask.expand_as(q)

        #k = self.k(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #4,8,4096,32
        #v = self.v(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 4,8,4096,32
        new_query=self.sr_kv(query.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, -1).permute(0, 2, 1)
        new_query = self.norm_kv(new_query)
        _,new_N,_ = new_query.shape
        kv=self.kv(new_query).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 4,8,256,32
        k, v = kv[0], kv[1]  # B,8,256,32
        attn_list=[]
        #re_query_list=[]
        for st_id in range(s_x.size(1)):  # shot 5
            if self.sr_ratio > 1:
                x_ = s_x[:,st_id,...].permute(0, 2, 1).reshape(B, C, H, W)#4,256,64,64
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) #4,256,256  #downsample
                x_ = self.norm(x_)#4,256,256
                q = self.q(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #8,8,4096,32 #8,8,256,32
            else:
                q = self.q(s_x[:,st_id,...]).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            #k, v = kv[0], kv[1] #4,8,256,32
            ##[4,8,N=256,32]*[4,8,4096,32]*=[4,8,4096,256]
            attn_ = (q @ k.transpose(-2, -1))#4,8,256,K_N=256
            attn = (attn_ * self.scale).softmax(dim=-1)#4,8,256,256  #,K_N=256维度加起来=1   support的数量是256
            # 4,8,4096,256  #4096维度加起来=1
            attn = self.attn_drop(attn)
            '''
            q2k_sim_idx = attn.max(3)[1]#4,8,256,
            q2k_sim_idx=q2k_sim_idx.unsqueeze(3).expand_as(q)  #4,8,4096，32  #np.array(q2k_sim_idx.data.cpu())
            re_query = torch.gather(q, 2, q2k_sim_idx)#4，8，4096，32
            #A=torch.gather(B,dim,index)从B里按照index把元素拿出来组成A
            re_query = re_query.transpose(1, 2).reshape(B, new_N, C)#B,256,256
            re_query_list.append(re_query)  ### v 4,8,4096,32
            '''
            x = (attn @ v).transpose(1, 2).reshape(B, new_N, C)  # 4,256,256
            x = self.proj(x)
            x = self.proj_drop(x)
            x=x.permute(0, 2, 1).reshape(B, C, H//self.sr_ratio, W//self.sr_ratio)
            x=F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True).reshape(B, C, -1).permute(0, 2, 1)
            attn_list.append(x)

        x = torch.stack(attn_list, dim=1)  # [8,5,256,256]
        x = torch.mean(x, 1)  # 4,2

        return x

class CyCTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384, 
                 num_heads=8, 
                 num_layers=2,
                 num_levels=1,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 shot=1,
                 rand_fg_num=300, 
                 rand_bg_num=300, 
                 ):
        super(CyCTransformer, self).__init__()
        self.embed_dims             = embed_dims
        self.num_heads              = num_heads
        self.num_layers             = num_layers
        self.num_levels             = num_levels
        self.num_points             = num_points
        self.use_ffn                = use_ffn
        self.feedforward_channels   = embed_dims*3
        self.dropout                = dropout
        self.shot                   = shot
        self.use_cross              = True
        self.use_self               = True
        self.use_cyc                = True
        
        self.rand_fg_num = rand_fg_num * shot
        self.rand_bg_num = rand_bg_num * shot

        if self.use_cross:
            self.cross_layers = []
        self.qry_self_layers  = []
        self.layer_norms = []
        self.ffns = []
        for l_id in range(self.num_layers):
            if self.use_cross:
                self.cross_layers.append(
                    MyCrossAttention(embed_dims, num_heads=12 if embed_dims%12==0 else self.num_heads, attn_drop=self.dropout, proj_drop=self.dropout,sr_ratio=4),
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))
                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
        for l_id in range(self.num_layers):
            if self.use_self:
                self.qry_self_layers.append(
                    MSDeformAttn(embed_dims, num_levels, 12 if embed_dims%12==0 else self.num_heads, num_points)
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))

                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))

        if self.use_cross: 
            self.cross_layers = nn.ModuleList(self.cross_layers)
        if self.use_self:
            self.qry_self_layers  = nn.ModuleList(self.qry_self_layers)
        if self.use_ffn:
            self.ffns         = nn.ModuleList(self.ffns)
        self.layer_norms  = nn.ModuleList(self.layer_norms)

        self.positional_encoding = SinePositionalEncoding(embed_dims//2, normalize=True) 
        self.level_embed = nn.Parameter(torch.rand(num_levels, embed_dims))
        nn.init.xavier_uniform_(self.level_embed)

        self.proj_drop  = nn.Dropout(dropout)
            

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

    def get_qry_flatten_input(self, x,prior_mask, qry_masks):
        src_flatten = [] 
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []
        prior_mask_flatten=[]
        for lvl in range(self.num_levels):   
            src = x[lvl]
            bs, c, h, w = src.shape
            
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1) # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)
            prior_mask_per=prior_mask[lvl]
            prior_mask_per=prior_mask_per.flatten(2).permute(0, 2, 1) #4,64*64,1
            prior_mask_flatten.append(prior_mask_per)
            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id]==255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w))

            pos_embed = self.positional_encoding(qry_valid_mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1) # [bs, num_elem, c]
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1) # [bs, num_elem]
        prior_mask_flatten=torch.cat(prior_mask_flatten,1) #4,4096,1
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1) # [bs, num_elem, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # [num_lvl, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [num_lvl]
        
        return src_flatten, prior_mask_flatten,qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index

    def get_supp_flatten_input(self, s_x, supp_mask, s_padding_mask):
        s_x_flatten = []
        supp_valid_mask = []
        supp_obj_mask = []  #s_padding_mask 4,5,512,512
        supp_mask = F.interpolate(supp_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) # [bs*shot, h, w] 4,5,64,64
        supp_mask = supp_mask.view(-1, self.shot, s_x.size(2), s_x.size(3)) #4,5,64,64

        s_padding_mask = F.interpolate(s_padding_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) # [bs*shot, h, w] 4,5,64,64
        s_padding_mask = s_padding_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))
        s_x = s_x.view(-1, self.shot, s_x.size(1), s_x.size(2), s_x.size(3)) #4,5,256,64,64

        for st_id in range(s_x.size(1)): #shot 5
            supp_valid_mask_s = []
            supp_obj_mask_s = []
            for img_id in range(s_x.size(0)): #bs 4
                supp_valid_mask_s.append(s_padding_mask[img_id, st_id, ...]==255)
                obj_mask = supp_mask[img_id, st_id, ...]==1
                if obj_mask.sum() == 0: # To avoid NaN
                    obj_mask[obj_mask.size(0)//2-1:obj_mask.size(0)//2+1, obj_mask.size(1)//2-1:obj_mask.size(1)//2+1] = True
                if (obj_mask==False).sum() == 0: # To avoid NaN
                    obj_mask[0, 0]   = False
                    obj_mask[-1, -1] = False 
                    obj_mask[0, -1]  = False
                    obj_mask[-1, 0]  = False
                supp_obj_mask_s.append(obj_mask) #64,64
            supp_valid_mask_s = torch.stack(supp_valid_mask_s, dim=0) # [bs, h, w]
            supp_valid_mask_s = supp_valid_mask_s.flatten(1) # [bs, h*w] #4,4096
            supp_valid_mask.append(supp_valid_mask_s) #4,4096

            supp_obj_mask_s = torch.stack(supp_obj_mask_s, dim=0) #4,64,64
            supp_obj_mask_s = (supp_obj_mask_s==1).flatten(1) # [bs, n] #4,4096
            supp_obj_mask.append(supp_obj_mask_s)

            s_x_s = s_x[:, st_id, ...]
            s_x_s = s_x_s.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c] 4,4096,256
            s_x_flatten.append(s_x_s.unsqueeze(1))#4,1,4096,256

        s_x_flatten = torch.cat(s_x_flatten, 1) # [bs, h*w*shot, c]
        supp_valid_mask = torch.cat(supp_valid_mask, 1)
        supp_mask_flatten = torch.cat(supp_obj_mask, 1)

        return s_x_flatten, supp_valid_mask, supp_mask_flatten

    def forward(self, x, prior_mask,qry_masks, s_x, supp_mask, s_padding_mask):
        if not isinstance(x, list): #4,256,64,64   #prior_mask, 8,1,,64,64
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]
        if not isinstance(prior_mask, list):
            prior_mask = [prior_mask.clone() for _ in range(self.num_levels)]
        assert len(x) == len(qry_masks) == self.num_levels
        bs, c,h,w = x[0].size()

        x_flatten,prior_mask_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(x,prior_mask, qry_masks)
        #4,4096,256; prior_mask_flatten=4,4096,1;4,4096;4,4096,256;1,2;1
        s_x, supp_valid_mask, supp_mask_flatten = self.get_supp_flatten_input(s_x, supp_mask.clone(), s_padding_mask.clone())
        ##4,5,4096,256;  4,20480,256;（20480=4096*5）  4,20480;  4,20480
        reference_points = self.get_reference_points(spatial_shapes, device=x_flatten.device)
        prior_mask_per = prior_mask[0].flatten(2).permute(0, 2, 1)
        #1,4096,1,2
        q = x_flatten
        pos = pos_embed_flatten

        ln_id = 0
        ffn_id = 0
        qry_feat=[]
        if self.use_cross:
            for l_id in range(self.num_layers):#2
                for st_id in range(5):
                    ln_id = 0
                    ffn_id = 0
                    cross_out = self.cross_layers[l_id](q,prior_mask_flatten, s_x,h,w) #list 5
                    q = cross_out + q
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1
                    if self.use_ffn:
                        q = self.ffns[ffn_id](q)
                        ffn_id += 1
                        q = self.layer_norms[ln_id](q)
                        ln_id += 1

        if self.use_self:
            for l_id in range(self.num_layers):  # 2
                q = q + self.proj_drop(self.qry_self_layers[l_id](q + pos, reference_points,q, spatial_shapes, level_start_index, qry_valid_masks_flatten))
                q = self.layer_norms[ln_id](q) #4，4096，256
                ln_id += 1
       
                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1
            qry_feat.append(q.permute(0, 2, 1).view(bs, c, h, w))


        #qry_feat = q.permute(0, 2, 1).view(bs, c, h, w) # [bs, c, num_ele]

        return qry_feat


