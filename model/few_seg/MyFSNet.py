import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm

import numpy as np
import random
import time
import cv2

from model.backbone.layer_extrator import layer_extrator
from model.backbone.transformer import SelfTransformer
from model.util.ASPP import ASPP, ASPP_Drop, ASPP_BN
from model.util.PSPNet import OneModel as PSPNet
from model.util.BaseNet import OneModel as BBaseNet
from torch.cuda.amp import autocast as autocast
import itertools

from collections import OrderedDict


def Cor_Map(query_feat, supp_feat_list, mask_list):
    # 支持分支经过mask和layer4的输出final_supp_list
    # query branch 经过layer4的输出query_feat
    corr_query_mask_list = []
    cosine_eps = 1e-7
    for i, tmp_supp_feat in enumerate(supp_feat_list):
        resize_size = tmp_supp_feat.size(2)  ##8,2048,64,64
        tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='nearest')

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask  # 再来经过mask帅选一遍
        q = query_feat  # 8,256,64,64
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)  # min-max normalization predict Mask
        # Prior Guided Feature Enrichment Network for Few-Shot Segmentation
        corr_query_mask_list.append(corr_query)
    corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
    return corr_query_mask


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class self_decode(nn.Module):
    def __init__(self, classes, embed_dims, num_layers, num_points):
        super(self_decode, self).__init__()

        self.classes = classes
        self.num_layers = num_layers

        self.init_merge = nn.Sequential(
            nn.Conv2d(embed_dims * 2 + 1, embed_dims, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.self_transformer = SelfTransformer(embed_dims=embed_dims, num_layers=self.num_layers,
                                                num_points=num_points)

        self.cls_aux = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(embed_dims, 1, kernel_size=1)
        )

    def forward(self, query_feat, supp_feat_bin, corr_query_mask_fg, padding_mask):
        merge_feat_bin = torch.cat([query_feat, supp_feat_bin, corr_query_mask_fg], 1)
        merge_feat_bin = self.init_merge(merge_feat_bin)  # 8,256,64,64
        fused_query_feat = self.self_transformer(merge_feat_bin, padding_mask.float())

        out_aux = self.cls_aux(fused_query_feat)

        return out_aux, fused_query_feat


class merge_decode(nn.Module):
    def __init__(self, classes, embed_dims, num_layers, num_points):
        super(merge_decode, self).__init__()

        self.classes = classes
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.merge_reduce = nn.Sequential(
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.self_transformer = SelfTransformer(embed_dims=self.embed_dims, num_layers=self.num_layers,
                                                num_points=num_points)

        self.merge_sum = nn.Sequential(
            # nn.GroupNorm(1, embed_dims, eps=1e-6),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self.embed_dims, self.classes, kernel_size=1)
        )

    def forward(self, query_feat_list, padding_mask):
        merge_query_feat = torch.cat(query_feat_list, 1)
        merge_query_feat = self.merge_reduce(merge_query_feat)

        x = self.self_transformer(merge_query_feat, padding_mask.float())
        x = self.merge_sum(x) + x

        out = self.cls(x)

        return out


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.shot = args.shot
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.sub_criterion =nn.NLLLoss(ignore_index=args.ignore_label)#nn.BCELoss(size_average=False)#
        self.pretrained = args.pretrain
        self.classes = 2
        self.fp16 = args.fp16
        self.backbone = args.backbone
        self.base_class_num = args.base_class_num
        if self.pretrained:
            BaseNet = BBaseNet(args)  # '/private/5-code/MyFewShot_res50/initmodel/PSPNet/{}/split{}/{}/best.pth
            weight_path ='E:/4-code/Base_FS_504_1_2/initmodel/BaseNet/{}/{}/split{}/best.pth'.format(args.dataset,args.backbone,args.split)
            # '/private/5-code/Base_FS_504_1_2/initmodel/BaseNet/{}/{}/split{}/best.pth'.format(
            #args.dataset, args.backbone, args.split) '/private/5-code/Base_FS_504_1_2/initmodel/BaseNet/{}/{}/split{}/best.pth'.format(args.dataset,args.backbone,args.split)
            # 'E:\\5-code\Base_FS_504_1_1\initmodel\BaseNet\iSAID\\resnet50\split0/best.pth'
            new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
            print('load <base> weights from: {}'.format(weight_path))
            for name, parameter in BaseNet.named_parameters():
                parameter.requires_grad_(False)
            try:
                BaseNet.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                BaseNet.load_state_dict(new_param)

            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = BaseNet.layer0, BaseNet.layer1, BaseNet.layer2, BaseNet.layer3, BaseNet.layer4
            self.res_1, self.res_2, self.res_3 = BaseNet.residual_block_1, BaseNet.residual_block_2, BaseNet.residual_block_3
            self.base_layer = nn.Sequential(BaseNet.ppm, BaseNet.cls)
        else:
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=args.backbone,
                                                                                             pretrained=True)

        reduce_dim = 256
        if self.backbone == 'vgg':
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query1 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp1 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_query2 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp2 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_query3 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp3 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_query4 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp4 = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.levels = 4
        self.self_transformer_decode_lvl = nn.ModuleList()
        self.num_layers = 1
        for i in range(self.levels):
            self.self_transformer_decode_lvl.append(
                self_decode(classes=self.classes, embed_dims=reduce_dim, num_layers=1,
                            num_points=9))
        # self.merge_reduce = nn.Linear(self.reduce_dim*4,self.reduce_dim)
        # self.dropout = nn.Dropout(0.1)

        self.merge_transformer_decode = merge_decode(classes=self.classes, embed_dims=reduce_dim,
                                                     num_layers=self.num_layers, num_points=9)

        self.iter = 0

    def forward(self, x, s_x, s_y, y, padding_mask=None, s_padding_mask=None, cat_idx=None):
        with autocast(enabled=self.fp16):
            x_size = x.size()
            bs = x_size[0]
            img_h = x_size[2]
            img_w = x_size[3]
            # Query Feature
            with torch.no_grad():
                query_feat_0 = self.layer0(x)  # 8,128,128,128,64,64
                query_feat_1 = self.layer1(query_feat_0)  # 8,256,128,128,64,64
                query_feat_2 = self.layer2(query_feat_1)  # 8,512,64,64
                query_feat_3 = self.layer3(query_feat_2)  # 8,1024,64,64

                query_res1 = self.res_1(query_feat_3)#8,1024,64,64
                query_res2 = self.res_2(query_res1)#8,1024,64,64
                query_res3 = self.res_3(query_res2)#8,1024,64,64

                query_feat_4 = self.layer4(query_res3)  # 8,2048,64,64
                query_out = self.base_layer(query_feat_4)  # 8,11,64,64
                query_out = nn.Softmax2d()(query_out)
                if self.backbone == 'vgg':
                    query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                                 mode='bilinear', align_corners=True)
            query_feat_tmp1 = self.down_query1(torch.cat([query_feat_3, query_feat_2], 1))  # 8,256,64,64
            query_feat_tmp2 = self.down_query2(torch.cat([query_res1, query_feat_2], 1))  # 8,256,64,64
            query_feat_tmp3 = self.down_query3(torch.cat([query_res2, query_feat_2], 1))  # 8,256,64,64
            query_feat_tmp4 = self.down_query4(torch.cat([query_res3, query_feat_2], 1))  # 8,256,64,64

            # query_feat_4_reduce=self.down_query4(query_feat_4)
            query_feat_bin_list = []
            query_feat_bin_list.append(query_feat_tmp1)
            query_feat_bin_list.append(query_feat_tmp2)
            query_feat_bin_list.append(query_feat_tmp3)
            query_feat_bin_list.append(query_feat_tmp4)

            h = query_feat_tmp1.size(2)
            w = query_feat_tmp1.size(3)

            # Support Feature
            final_supp_list = []
            mask_list_fg = []
            mask_list_bg = []
            pro_fg_list = []
            pro_fg_list_1 = []
            pro_fg_list_2 = []
            pro_fg_list_3 = []
            mid_supp_feat_list = []
            for i in range(self.shot):

                mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)  # 8,1,512,512
                mask_list_fg.append(mask)
                mask_list_bg.append(1 - mask)
                with torch.no_grad():
                    supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                    supp_feat_1 = self.layer1(supp_feat_0)
                    supp_feat_2 = self.layer2(supp_feat_1)
                    supp_feat_3 = self.layer3(supp_feat_2)  # 8,1024,64,64

                    supp_res1 = self.res_1(supp_feat_3)
                    supp_res2 = self.res_2(supp_res1)
                    supp_res3 = self.res_3(supp_res2)

                    supp_feat_4_true = self.layer4(supp_res3)  # 8,2048,64,64
                    mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                         mode='nearest')  # 8,1,64,64
                    supp_feat_4 = self.layer4(supp_res3 * mask)  # 8,2048,64,64
                    final_supp_list.append(supp_feat_4)
                    if self.backbone == 'vgg':
                        supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                    mode='bilinear', align_corners=True)

                    supp_base_out = self.base_layer(supp_feat_4_true.clone())
                    supp_base_out = nn.Softmax2d()(supp_base_out)  # b*(c+1)*h*w #8,11,64,64

                supp_feat_tmp1 = self.down_supp1(torch.cat([supp_feat_3, supp_feat_2], 1))  # 8,256,64,64
                supp_feat_tmp2 = self.down_supp2(torch.cat([supp_res1, supp_feat_2], 1))  # 8,256,64,64
                supp_feat_tmp3 = self.down_supp3(torch.cat([supp_res2, supp_feat_2], 1))  # 8,256,64,64
                supp_feat_tmp4 = self.down_supp4(torch.cat([supp_res3, supp_feat_2], 1))  # 8,256,64,64

                # gen pro#
                pro_fg = Weighted_GAP(supp_feat_tmp1, mask)  # 8,256,1,1
                pro_fg_list.append(pro_fg)  # fg supp_feat_4_reduce

                pro_fg_1 = Weighted_GAP(supp_feat_tmp2, mask)  # 8,256,1,1
                pro_fg_list_1.append(pro_fg_1)  # fg
                pro_fg_2 = Weighted_GAP(supp_feat_tmp3, mask)  # 8,256,1,1
                pro_fg_list_2.append(pro_fg_2)  # fg
                pro_fg_3 = Weighted_GAP(supp_feat_tmp4, mask)  # 8,256,1,1
                pro_fg_list_3.append(pro_fg_3)  # fg

            supp_feat_bin_0 = torch.mean(torch.stack(pro_fg_list, dim=1), 1).expand(-1, -1, h, w)  # 4,256,1,1
            supp_feat_bin_1 = torch.mean(torch.stack(pro_fg_list_1, dim=1), 1).expand(-1, -1, h, w)  # 4,256,1,1
            supp_feat_bin_2 = torch.mean(torch.stack(pro_fg_list_2, dim=1), 1).expand(-1, -1, h, w)  # 4,256,1,1
            supp_feat_bin_3 = torch.mean(torch.stack(pro_fg_list_3, dim=1), 1).expand(-1, -1, h, w)  # 4,256,1,1
            supp_feat_bin_list = []
            supp_feat_bin_list.append(supp_feat_bin_0)
            supp_feat_bin_list.append(supp_feat_bin_1)
            supp_feat_bin_list.append(supp_feat_bin_2)
            supp_feat_bin_list.append(supp_feat_bin_3)

            # 支持分支经过mask和layer4的输出final_supp_list
            corr_query_mask_fg = Cor_Map(query_feat_4, final_supp_list, mask_list_fg)  # 8,1,64,64
            output_fin_list = []
            multi_lvl_query_feat_list = []
            # with torch.no_grad():
            for i in range(self.levels):
                out_aux, fused_query_feat = self.self_transformer_decode_lvl[i](query_feat_bin_list[i],
                                                                                supp_feat_bin_list[i],
                                                                                corr_query_mask_fg, padding_mask)

                multi_lvl_query_feat_list.append(fused_query_feat)
                output_fin_aux = F.interpolate(out_aux, size=(img_h, img_w), mode='bilinear', align_corners=True)
                output_fin_list.append(output_fin_aux)

            out = self.merge_transformer_decode(multi_lvl_query_feat_list, padding_mask)

            output_fin = F.interpolate(out, size=(img_h, img_w), mode='bilinear', align_corners=True)
            # output_fin_list.append(output_fin)

            s1 = output_fin_list[0]  #foreground 8,1,512,512  nn.Sigmoid()
            s2 = output_fin_list[1]
            s3 = output_fin_list[2]
            s4 = output_fin_list[3]

            p1_1 = s1
            p2_1 = s2 + s1
            p3_1 = s3 + s2 + s1
            p4_1 = s4 + s3 + s2 + s1
            p2_2 = s2 + s3 + s4
            p3_2 = s3 + s4
            p4_2 = s4

            probs_list = []
            probs_list.append(p1_1)
            probs_list.append(p2_1)
            probs_list.append(p3_1)
            probs_list.append(p4_1)

            probs_list.append(p2_2)
            probs_list.append(p3_2)
            probs_list.append(p4_2)


            if self.training:

                act_map = nn.Softmax(1)(output_fin)  # 8,2,512,512
                act_map = act_map[:, 1]  # 8,512,512,===8,1,512,512
                alpha = self.GAP(act_map.unsqueeze(1))  # 8,1,1,1 前景占整个image的比例
                main_loss = self.criterion(output_fin, y.long())

                mask_y = (y == 1).float().unsqueeze(1)
                alpha_1 = self.GAP(mask_y)  # r
                beta = (alpha - alpha_1) ** 2
                aux_loss = -(1 - alpha) * torch.log(alpha) - beta * torch.log(1 - beta)
                aux_loss = torch.mean(aux_loss)

                
                aux_loss_all = 0.0
                main_loss_all = 0.0
                for i in range(len(probs_list)):
                    prob = nn.Sigmoid()(probs_list[i])  # 8,1,512,512

                    sub_y=y.unsqueeze(1)
                    weights = torch.ones(bs, 1, img_h, img_w).cuda()
                    weights[sub_y == 255] = 0
                    aux_main_loss = nn.BCELoss(weights)(prob, sub_y.float())
                    main_loss_all = main_loss_all + aux_main_loss

                    act_map = prob#[:, 1]  # 8,512,512,===8,1,512,512
                    alpha = self.GAP(act_map)  #.unsqueeze(1) 8,1,1,1 前景占整个image的比例
                    mask_y = (y == 1).float().unsqueeze(1)
                    alpha_1 = self.GAP(mask_y)  # r
                    beta = (alpha - alpha_1) ** 2
                    aux_aux_loss = -(1 - alpha) * torch.log(alpha) - beta * torch.log(1 - beta)
                    aux_aux_loss = torch.mean(aux_aux_loss)
                    aux_loss_all = aux_loss_all + aux_aux_loss
                #   aux seg support------------------
                aux_main_loss = main_loss_all / len(probs_list)

                aux_aux_loss = aux_loss_all / len(probs_list)

                
                return output_fin.max(1)[1], main_loss, aux_loss, aux_main_loss, aux_aux_loss
            else:
                return output_fin, output_fin_list

