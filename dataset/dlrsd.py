r""" iSAID-5i few-shot semantic segmentation dataset """
import os
from typing_extensions import override

import torch
import PIL.Image as Image
import numpy as np
#import torchvision.transforms.v2 as transforms2
#from torchvision import transforms  data.
from .transforms import *

from .pascal import DatasetPASCAL,BaseDatasetPASCAL


class DatasetDLRSD(DatasetPASCAL):
    def __init__(self, datapath,fold, transform, mode, shot, use_original_imgsize, aug) -> None:
        self.mode = 'val' if mode in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'dlrsd'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.class_name = {
            0: 'unlabeled',
            1: 'airplane',
            2: 'baresoil',
            3: 'buildings',
            4: 'cars',
            5: 'chaparral',
            6: 'court',
            7: 'dock',
            8: 'field',
            9: 'grass',
            10: 'mobilehome',
            11: 'pavement',
            12: 'sand',
            13: 'sea',
            14: 'ship',
            15: 'tanks',
        }

        self.img_path = os.path.join(datapath, 'UCMerced_LandUse/Images')
        self.ann_path = os.path.join(datapath, 'DLRSD/Images')

        self.aug = aug and (self.mode == 'trn')
        if self.aug:
            self.tv2 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                # transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0))
            ])

        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    @override
    def __len__(self):
        return len(self.img_metadata)  # TODO: why hsnet use 100 for val

    @override
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        mask = Image.open(os.path.join(self.ann_path, img_name[:-2], img_name + '.png'))
        return mask

    @override
    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name[:-2], img_name) + '.tif')

    @override
    def build_img_metadata(self):
    
        def read_metadata(mode, fold_id):
            fold_n_metadata = os.path.join('data/splits/DLRSD/%s/fold%d.txt' % (mode, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1])] for data in fold_n_metadata]
            #print(fold_n_metadata)
            return fold_n_metadata

        img_metadata = []
        if self.mode == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        elif self.mode == 'val': # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.mode, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.mode)

        print('Total (%s) images are : %d' % (self.mode, len(img_metadata)))

        return img_metadata


class BaseDatasetDLRSD(BaseDatasetPASCAL):
    def __init__(self, datapath,fold, transform, mode, use_original_imgsize, aug) -> None:
        self.mode = 'val' if mode in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'dlrsd'
        self.use_original_imgsize = use_original_imgsize


        self.img_path = os.path.join(datapath, 'UCMerced_LandUse/Images')
        self.ann_path = os.path.join(datapath, 'DLRSD/Images')

        self.aug = aug and (self.mode == 'trn')
        if self.aug:
            self.tv2 = Compose([
                RandomHorizontalFlip(),
                RandRotate([-10,10], padding=[123.675, 116.28, 103.53], ignore_label=255),
                # transforms2.RandomResizedCrop(size=256, scale=(0.5, 1.0))
            ])
        '''
        train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
        
            rotate: [-10,10]  #minimum and maximum random rotate
            padding: [123.675, 116.28, 103.53]
            ignore_label: 255
        '''
        self.transform = transform
        self.class_name = {
            0: 'unlabeled',
            1: 'airplane',
            2: 'baresoil',
            3: 'buildings',
            4: 'cars',
            5: 'chaparral',
            6: 'court',
            7: 'dock',
            8: 'field',
            9: 'grass',
            10: 'mobilehome',
            11: 'pavement',
            12: 'sand',
            13: 'sea',
            14: 'ship',
            15: 'tanks',
            #16:'trees',
            #17:'water'
        }
        '''
        self.class_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        if self.fold == 2:
            self.class_ids = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
        elif self.fold == 1:
            self.class_ids = list(range(1, 6)) + list(range(11, 16))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
        elif self.fold == 0:
            self.class_ids = list(range(6, 16))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]
        '''
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    @override
    def __len__(self):
        return len(self.img_metadata)  # TODO: why hsnet use 100 for val

    @override
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        mask = Image.open(os.path.join(self.ann_path, img_name[:-2], img_name + '.png'))
        return mask

    @override
    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name[:-2], img_name) + '.tif')

    @override
    def build_img_metadata(self):

        def read_metadata(mode, fold_id):
            fold_n_metadata = os.path.join('data/splits/DLRSD/%s/fold%d.txt' % (mode, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('_')[0], int(data.split('_')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.mode == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        elif self.mode == 'val': # For validation, read image-metadata of "current" fold
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        else:
            raise Exception('Undefined split %s: ' % self.mode)

        print('Total (%s) images are : %d' % (self.mode, len(img_metadata)))

        return img_metadata
