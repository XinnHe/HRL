import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        
from model.backbone.transformer import SelfTransformer
from model.backbone.layer_extrator import layer_extrator
from torch.cuda.amp import autocast
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class ResidualBottleneck(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1,expansion=2, downsample=None):
        super(ResidualBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.classes = args.base_class_num +1  #10+1
        self.backbone = args.backbone

        self.fp16 = args.fp16

        if args.backbone in ['vgg', 'resnet50', 'resnet101']:
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=args.backbone, pretrained = True)

            fea_dim = 512 if args.backbone == 'vgg' else 2048
            reduce_dim = 512 if args.backbone == 'vgg' else 1024  # 256
            expansion = 2 if args.backbone == 'vgg' else 4  # 256


        self.residual_block_1 = ResidualBottleneck(inplanes=reduce_dim, planes=256,expansion=expansion)
        self.residual_block_2 = ResidualBottleneck(inplanes=reduce_dim, planes=256,expansion=expansion)
        self.residual_block_3 = ResidualBottleneck(inplanes=reduce_dim, planes=256,expansion=expansion)
        # Base Learner
        bins = (1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.classes, kernel_size=1))

    def forward(self, x, y,padding_mask):
        with autocast(enabled=self.fp16):
            x_size = x.size()
            h = x_size[2]
            w = x_size[3]
            x_0 = self.layer0(x)  # 8,128,128,128
            x_1 = self.layer1(x_0)  # 8,256,128,128
            x_2 = self.layer2(x_1)  # 8,512,64,64
            x_3 = self.layer3(x_2)  # 8,1024,64,64
            #x_3_reduce=self.channel_reduce(x_3) #1024--512
            '''
            x_att1 = self.self_transformer_1(x_3_reduce,padding_mask.float())
            x_att2 = self.self_transformer_2(x_att1,padding_mask.float())
            x_att3 = self.self_transformer_3(x_att2,padding_mask.float())
            '''

            x_att1 = self.residual_block_1(x_3)
            x_att2 = self.residual_block_2(x_att1)
            x_att3 = self.residual_block_2(x_att2)

            #x_3_up=self.channel_up(x_att3)
            x_4 = self.layer4(x_att3)  # 8,2048,64,64

            x = self.ppm(x_4)

            out = self.cls(x)

            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

            if self.training:
                main_loss = self.criterion(out, y.long())
                return out.max(1)[1], main_loss, 0, 0
            else:
                return out