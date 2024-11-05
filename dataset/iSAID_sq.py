from lib2to3.pgen2.token import N_TOKENS
import os
import os.path as osp
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
from PIL import Image

from util.util import make_dict, gen_list
from util.get_transform import get_transform
from util.get_weak_anns import transform_anns


def label_trans(label, target_cls, mode):
    label_class = np.unique(label).tolist()
    if 0 in label_class:
        label_class.remove(0)

    target_pix = np.where(label == target_cls)
    new_label = np.zeros_like(label)
    new_label[target_pix[0], target_pix[1]] = 1  # 把选中的_cls 位置变为1---
    # new_label是0，1mask

    if mode == 'ignore':
        for cls in label_class:
            if cls != target_cls:  # 把没有选中的位置设为255
                ignore_pix = np.where(label == cls)
                new_label[ignore_pix[0], ignore_pix[1]] = 255
    elif mode == 'To_zero':
        ignore_pix = np.where(label == 255)  # 保持mask里255
        new_label[ignore_pix[0], ignore_pix[1]] = 255

    return new_label



class vis_Few_Data(Dataset):
    class_id = None
    all_class = None
    val_class = None

    data_root = None
    val_list = None
    train_list = None

    def __init__(self, split=0, shot=1, dataset=None, mode='train', ann_type='mask', transform_dict=None,
                 ori_resize=False):

        assert mode in ['train', 'val', 'demo']

        self.mode = mode
        self.shot = shot
        self.ann_type = ann_type

        self.sample_mode = transform_dict.pop('sample_mode')
        self.fliter_mode = transform_dict.pop('fliter_mode')

        if self.mode == 'train':
            self.list = list(set(self.all_class) - set(self.val_class[split]))
        else:
            self.list = self.val_class[split]

        self.subcls_list=[]
        self.support_list=[]
        self.query_list=[]
        self.data_list=[]
        fold_n_metadata="/private/5-code/Base_FS_521_2_1/visexp/vis_MyFSNet_iSAID/resnet50_split0_1shot/2024-10-01-01-24-20-214365/support_query.txt"
        with open(fold_n_metadata, 'r') as f:
            #self.subcls_list,self.support_list,self.query_list = f.read().split('--')[0],f.read().split('--')[1],f.read().split('--')[2]#[:-1]data.split('__')
            self.data_list = f.read().split('\n')
        #
        self.transform_dict = transform_dict
        self.AUG = get_transform(transform_dict)
        self.class_list = []

    def transform(self, image, label, padding_mask=None):
        if self.transform_dict['type'] == 'albumentations':
            aug = self.AUG(image=image, mask=label)
            return aug['image'], aug['mask']
        else:
            image, label, padding_mask = self.AUG(image=image, label=label, padding_mask=padding_mask)
            return image, label, padding_mask

    def count(self, ):
        for j in range(len(self.list)):
            print(j, self.class_list.count(j))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        item=self.data_list[index]
        #print('item: ',item)
        class_chosen_index=int(item.split("--")[0])
        
        class_chosen=self.list[class_chosen_index]
        
        support_img=item.split("--")[1]
        #print(support_img)
        query_img=item.split("--")[2]
        #print(query_img)
        
        
        
        image_path = query_img
        label_path='../data/iSAID/ann_dir/val/'+query_img.split('/')[-1][:-4]+'_instance_color_RGB.png'
        #print('label_path: ',label_path)
        

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # from train\ read a file
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        padding_mask = np.zeros_like(label)
        
        query_name=image_path
        ori_img=cv2.imread(image_path, cv2.IMREAD_COLOR) # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#
        
        label_class = np.unique(label).tolist()  #
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        label = label_trans(label, class_chosen, mode='To_zero')  # 没选中的是0，选中的变1，255保持
        
        
        if True:
            support_name=[]
            support_image_list_ori = []
            support_label_list_ori = []
            support_label_list_ori_mask = []
            support_padding_list_ori = []
            subcls_list = []
            for k in range(self.shot):
                # subcls_list=list[3]  #self.list [6,7,8,9,10,11,12,13,14,15] print('------hhisn---------')
                subcls_list.append(class_chosen_index)  # class_chosen9
               
                support_image_path =support_img
                support_label_path ='../data/iSAID/ann_dir/val/'+support_img.split('/')[-1][:-4]+'_instance_color_RGB.png'
                support_name.append(support_image_path)
                
                support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
                support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
                support_image = np.float32(support_image)
                
                support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
                support_label = label_trans(support_label, class_chosen, mode='To_zero')
                # 

                support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # self.ann_type=mask
                support_label_mask = label_trans(support_label_mask, class_chosen, mode='To_zero')
                
                support_padding_label = np.zeros_like(support_label)
                support_padding_label[support_label == 255] = 255

                if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                    raise (RuntimeError(
                        "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
                support_image_list_ori.append(support_image)
                support_label_list_ori.append(support_label)
                support_label_list_ori_mask.append(support_label_mask)
                support_padding_list_ori.append(support_padding_label)
            assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot

            raw_image = image.copy()
            raw_label = label.copy()
            support_image_list = [[] for _ in range(self.shot)]
            support_label_list = [[] for _ in range(self.shot)]
            support_padding_list = [[] for _ in range(self.shot)]
            image, label, padding_mask = self.transform(image, label, padding_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k], support_padding_list[k] = self.transform(
                    support_image_list_ori[k], support_label_list_ori[k], support_padding_list_ori[k])

            s_xs = support_image_list  # list
            s_ys = support_label_list
            

            s_x = s_xs[0].unsqueeze(0)  # [self.shot,3,512,512]
            for i in range(1, self.shot):
                s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
            s_y = s_ys[0].unsqueeze(0)
            for i in range(1, self.shot):
                s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

            if support_padding_list is not None:
                s_eys = support_padding_list
                s_ey = s_eys[0].unsqueeze(0)
                for i in range(1, self.shot):
                    s_ey = torch.cat([s_eys[i].unsqueeze(0), s_ey], 0)

            total_image_list = support_image_list_ori.copy()
            total_image_list.append(raw_image)
            
        
        # Return  Query=image
        if self.mode == 'train':
            return image, label, s_x, s_y, padding_mask, s_ey, subcls_list
        elif self.mode == 'val':
            return image, label, s_x, s_y, padding_mask, s_ey, subcls_list, raw_label,ori_img,support_name,query_name
        elif self.mode == 'demo':
            return image, label, s_x_list, s_y_list, subcls_list, s_ori_x_list, s_ori_y_list, raw_image, raw_label



class iSAID_vis_few_dataset(vis_Few_Data):

    class_id = {
                0: 'unlabeled',
                1: 'ship',
                2: 'storage_tank',
                3: 'baseball_diamond',  
                4: 'tennis_court',
                5: 'basketball_court',
                6: 'Ground_Track_Field',
                7: 'Bridge',
                8: 'Large_Vehicle',
                9: 'Small_Vehicle',
                10: 'Helicopter',
                11: 'Swimming_pool',
                12: 'Roundabout',
                13: 'Soccer_ball_field',
                14: 'plane',
                15: 'Harbor'
                    }
    
    PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
               [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
               [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
               [0, 127, 191], [0, 127, 255], [0, 100, 155]]
    
    all_class = list(range(1, 16))
    val_class = [list(range(1, 6)), list(range(6, 11)), list(range(11, 16))]
    #[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    data_root = '../data/iSAID'
    train_list ='./lists/iSAID/train.txt'
    val_list ='./lists/iSAID/val.txt'

    def __init__(self, split=0, shot=1, dataset='iSAID', mode='train', ann_type='mask', transform_dict=None):
        super().__init__(split, shot, dataset, mode, ann_type, transform_dict)

