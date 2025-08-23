r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import random

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, mode, shot, use_original_imgsize):
        self.mode = 'val' if mode in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
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
        }
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        #print('**************************class*************************************')
    def __len__(self):
        return len(self.img_metadata)# if self.mode == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000'''
        
        if self.mode == 'trn':
            
            class_sample = random.choice(self.class_ids)
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            query_name, class_sample = self.img_metadata[idx]

        support_names= self.sample_episode_support(query_name, class_sample)
        
        '''
        #print('**************************class*************************************')
        #class_sample = random.choice(self.class_ids)
        #query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        #support_names= self.sample_episode_support(query_name, class_sample)
        '''
        #query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)
    
        if self.aug:
            query_img, query_cmask = self.tv2(query_img, query_cmask)
            for i in range(self.shot):
                support_imgs[i], support_cmasks[i] = self.tv2(support_imgs[i], support_cmasks[i])

        query_cmask = torch.tensor(np.array(query_cmask))
        for i in range(self.shot):
            support_cmasks[i] = torch.tensor(np.array(support_cmasks[i]))

        org_qry_img=torch.tensor(np.array(query_img.resize((256,256))))
        
        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        query_mask=query_mask.long()
        padding_mask = torch.zeros_like(query_mask)
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask.long())
            support_ignore_idxs.append(support_ignore_idx)

        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)
        #print('---',class_sample)
        class_sample_index=self.class_ids.index(class_sample)
        #print(class_sample,class_sample_index)
        #print('---00000',class_sample_index)
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'query_padding_mask': padding_mask,
                 'org_query_imsize': org_qry_imsize,
                 'org_query_image': org_qry_img,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,
                 'class_sample_index': torch.tensor(class_sample_index),
                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id] = 0
        mask[mask == class_id] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def sample_episode_support(self, query_name, class_sample):
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break
        return support_names
        
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i+1 for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass+1) if x not in class_ids_val]
        class_ids_trn.remove(0)
        if self.mode == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(mode, fold_id):
            fold_n_metadata = os.path.join('data/splits/pascal/%s/fold%d.txt' % (mode, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1])] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.mode == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        elif self.mode == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.mode, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.mode)

        print('Total (%s) images are : %d' % (self.mode, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(1, self.nclass+1):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            #print(img_name, img_class)
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise


class BaseDatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, mode, use_original_imgsize):
        self.mode = 'val' if mode in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'

        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
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
        self.class_ids =self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.mode == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        img_name, class_sample = self.img_metadata[idx]

        img, mask, org_imsize = self.load_frame(img_name)#PIL
        #img, mask, org_imsize = self.load_frame('airplane80') #'E:/5-code\data/remote_sensing/DLRSD/Images/airplane/airplane80'
        padding_mask = np.zeros_like(mask)
        if self.aug:
            img, mask,padding_mask = self.tv2(img, mask,padding_mask)
        #mask = torch.tensor(np.array(mask))

        mask=np.array(mask)
        img = self.transform(img)
        #img = np.float32(img)
        if not self.use_original_imgsize:
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), img.size()[-2:],
                                        mode='nearest').squeeze()

        #mask, ignore_idx = self.extract_ignore_idx(mask.float(), class_sample)
        label_tmp = mask.copy() #.clone()#
        label_class = np.unique(mask).tolist()
        if 255 in label_class:
            label_class.remove(255)
        if 0 in label_class:
            label_class.remove(0)

        for cls in label_class:
            select_pix = np.where(label_tmp == cls)
            if cls in self.class_ids:
                mask[select_pix[0], select_pix[1]] = self.class_ids.index(cls) + 1
            #elif cls>self.nclass:
            #    mask[select_pix[0], select_pix[1]] =255
            else:
                mask[select_pix[0], select_pix[1]] = 0
        #print(mask.shape,img.shape,img_name)
        mask = torch.tensor(np.array(mask)).long()
        #img = torch.tensor(np.array(img))
        batch = {'img': img,
                 'mask': mask,
                 'img_name': img_name,
                 #'ignore_idx': ignore_idx,
                 'org_imsize': org_imsize,
                 'class_id': torch.tensor(class_sample)}
        #batch
        return batch#img, mask

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, img_name):
        img = self.read_img(img_name)
        mask = self.read_mask(img_name)

        org_imsize = img.size

        return img, mask, org_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')


    #'''
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds #5
        class_ids_val = [self.fold * nclass_trn + i+1 for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass+1) if x not in class_ids_val]
        class_ids_trn.remove(0)
        if self.mode == 'trn':
            return class_ids_trn
        else:
            return class_ids_trn
    #'''

    def build_img_metadata(self):

        def read_metadata(mode, fold_id):
            fold_n_metadata = os.path.join('data/splits/pascal/%s/fold%d.txt' % (mode, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.mode == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        elif self.mode == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.mode, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.mode)

        print('Total (%s) images are : %d' % (self.mode, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
