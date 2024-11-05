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

        if self.mode == 'demo':
            dict_name = './lists/{}/train_dict.txt'.format(dataset)
            self.sample_mode = 'class'
        else:
            dict_name = './lists/{}/{}_dict.txt'.format(dataset, mode)
        if not os.path.exists(dict_name):
            make_dict(data_root=self.data_root, data_list=eval('self.{}_list'.format(self.mode)), \
                      all_class=self.all_class, dataset=dataset, mode=self.mode)
        # self.fliter_mode=True
        self.data_list, self.sub_class_file_list = gen_list(dict_name, self.list, fliter=self.fliter_mode)
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
        if self.sample_mode == 'rand':
            image_path, label_path = self.data_list[index]  # train class 所有图 10813，split0
        elif self.sample_mode == 'class':
            tmp_class = self.list[random.randint(1, len(self.list)) - 1]
            file_all = list(set(self.sub_class_file_list[tmp_class]) & set(self.data_list))
            image_path, label_path = file_all[random.randint(1, len(file_all)) - 1]
            class_chosen = tmp_class
            
            

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

        new_label_class = []
        for c in label_class:
            if c in self.list:  # self.list=[6,7,8,9,10,11,12,13,14,15]
                new_label_class.append(c)

        label_class = new_label_class
        assert len(label_class) > 0

        # supp_query dataset
        if self.sample_mode == 'rand':
            class_chosen = label_class[random.randint(1, len(label_class)) - 1]

        label = label_trans(label, class_chosen, mode='To_zero')  # 没选中的是0，选中的变1，255保持

        file_class_chosen = self.sub_class_file_list[class_chosen]  # 选出包含类别9的所有图的list
        num_file = len(file_class_chosen)  # 5181 选出5181张有类别9的图

        if self.mode == 'demo':
            s_x_list = []
            s_y_list = []
            s_ori_x_list = []
            s_ori_y_list = []

            raw_image = image.copy()
            raw_label = label.copy()
            
        else:
            support_image_path_list = []
            support_label_path_list = []
            support_idx_list = []
            support_name=[]
            
            for k in range(self.shot):
                support_idx = random.randint(1, num_file) - 1
                support_image_path = image_path
                support_label_path = label_path
                while ((
                               support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                    # 取5shot次图，保证每次不一样 保证与query（=image，一开始取的图）不同
                    support_idx = random.randint(1, num_file) - 1
                    support_image_path, support_label_path = file_class_chosen[support_idx]
                support_idx_list.append(support_idx)
                support_image_path_list.append(support_image_path)
                support_label_path_list.append(support_label_path)

            support_image_list_ori = []
            support_label_list_ori = []
            support_label_list_ori_mask = []
            support_padding_list_ori = []
            subcls_list = []
            for k in range(self.shot):
                # subcls_list=list[3]  #self.list [6,7,8,9,10,11,12,13,14,15]
                subcls_list.append(self.list.index(class_chosen))  # class_chosen9的索引
                support_image_path = support_image_path_list[k]
                support_label_path = support_label_path_list[k]
                support_name.append(support_image_path)
                support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
                support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
                support_image = np.float32(support_image)

                support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
                support_label = label_trans(support_label, class_chosen, mode='To_zero')
                # 选中的cls变成1，其他变成0，255不变

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

class Few_Data(Dataset):
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
            
        #print('00000',len(self.list))

        if self.mode == 'demo':
            dict_name = './lists/{}/train_dict.txt'.format(dataset)
            self.sample_mode = 'class'
        else:
            dict_name = './lists/{}/{}_dict.txt'.format(dataset, mode)
        if not os.path.exists(dict_name):
            make_dict(data_root=self.data_root, data_list=eval('self.{}_list'.format(self.mode)), \
                      all_class=self.all_class, dataset=dataset, mode=self.mode)
        # self.fliter_mode=True
        self.data_list, self.sub_class_file_list = gen_list(dict_name, self.list, fliter=self.fliter_mode)
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
        if self.sample_mode == 'rand':
            image_path, label_path = self.data_list[index]  # train class 所有图 10813，split0
        elif self.sample_mode == 'class':
            tmp_class = self.list[random.randint(1, len(self.list)) - 1]
            file_all = list(set(self.sub_class_file_list[tmp_class]) & set(self.data_list))
            image_path, label_path = file_all[random.randint(1, len(file_all)) - 1]
            class_chosen = tmp_class

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # from train\ read a file
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        padding_mask = np.zeros_like(label)

        label_class = np.unique(label).tolist()  #
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []
        for c in label_class:
            if c in self.list:  # self.list=[6,7,8,9,10,11,12,13,14,15]
                new_label_class.append(c)

        label_class = new_label_class
        assert len(label_class) > 0

        # supp_query dataset
        if self.sample_mode == 'rand':
            class_chosen = label_class[random.randint(1, len(label_class)) - 1]

        label = label_trans(label, class_chosen, mode='To_zero')  # 没选中的是0，选中的变1，255保持

        file_class_chosen = self.sub_class_file_list[class_chosen]  # 选出包含类别9的所有图的list
        num_file = len(file_class_chosen)  # 5181 选出5181张有类别9的图

        if self.mode == 'demo':
            s_x_list = []
            s_y_list = []
            s_ori_x_list = []
            s_ori_y_list = []

            raw_image = image.copy()
            raw_label = label.copy()
        else:
            support_image_path_list = []
            support_label_path_list = []
            support_idx_list = []
            for k in range(self.shot):
                support_idx = random.randint(1, num_file) - 1
                support_image_path = image_path
                support_label_path = label_path
                while ((
                               support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                    # 取5shot次图，保证每次不一样 保证与query（=image，一开始取的图）不同
                    support_idx = random.randint(1, num_file) - 1
                    support_image_path, support_label_path = file_class_chosen[support_idx]
                support_idx_list.append(support_idx)
                support_image_path_list.append(support_image_path)
                support_label_path_list.append(support_label_path)

            support_image_list_ori = []
            support_label_list_ori = []
            support_label_list_ori_mask = []
            support_padding_list_ori = []
            subcls_list = []
            for k in range(self.shot):
                # subcls_list=list[3]  #self.list [6,7,8,9,10,11,12,13,14,15]
                subcls_list.append(self.list.index(class_chosen))  # class_chosen9的索引
                support_image_path = support_image_path_list[k]
                support_label_path = support_label_path_list[k]
                support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
                support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
                support_image = np.float32(support_image)

                support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
                support_label = label_trans(support_label, class_chosen, mode='To_zero')
                # 选中的cls变成1，其他变成0，255不变

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
            return image, label, s_x, s_y, padding_mask, s_ey, subcls_list, raw_label
        elif self.mode == 'demo':
            return image, label, s_x_list, s_y_list, subcls_list, s_ori_x_list, s_ori_y_list, raw_image, raw_label


class Base_Data(Dataset):
    class_id = None
    all_class = None
    val_class = None

    data_root = None
    val_list = None
    train_list = None

    def __init__(self, split=0, data_root=None, dataset=None, mode='train', transform_dict=None):

        assert mode in ['train', 'val']

        self.mode = mode

        self.transform_dict = transform_dict
        self.AUG = get_transform(transform_dict)

        if split == -1:
            self.list = self.all_class
        else:
            self.list = list(set(self.all_class) - set(self.val_class[split]))

        dict_name = './lists/{}/{}_dict.txt'.format(dataset, mode)
        if not os.path.exists(dict_name):
            make_dict(data_root=self.data_root, data_list=eval('self.{}_list'.format(self.mode)), \
                      all_class=self.all_class, dataset=dataset, mode=self.mode)

        self.data_list, _ = gen_list(dict_name, self.list, fliter=False)

        # if split == -1 :
        #     self.list = self.all_class
        #     list_path = './lists/{}/{}.txt'.format(dataset, self.mode)
        #     with open(list_path, 'r') as f:
        #         f_str = f.readlines()
        #     self.data_list = []
        #     for line in f_str:
        #         img, mask = line.split(' ')
        #         img = '../data/{}/'.format(dataset) + img
        #         mask = '../data/{}/'.format(dataset) + mask
        #         self.data_list.append((img, mask.strip()))
        # else:
        #     self.list = list(set(self.all_class) - set(self.val_class[split]))
        #     list_root = './lists/{}/fss_list/{}/'.format(dataset, self.mode)
        #     # dict_path = list_root + '{}_dict.txt'.format(self.mode)

        #     if self.mode == 'train':
        #         list_path = list_root + 'train_split{}.txt'.format(split)
        #     elif self.mode == 'val':
        #         list_path = list_root + 'val_base{}.txt'.format(split)

        #     with open(list_path, 'r') as f:
        #         f_str = f.readlines()
        #     self.data_list = []
        #     for line in f_str:
        #         img, mask = line.split(' ')
        #         self.data_list.append((img, mask.strip()))


    def transform(self, image, label, padding_mask=None):
        if self.transform_dict['type'] == 'albumentations':
            aug = self.AUG(image=image, mask=label)
            return aug['image'], aug['mask']
        else:
            image, label, padding_mask = self.AUG(image=image, label=label, padding_mask=padding_mask)
            return image, label, padding_mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_tmp = label.copy()
        padding_mask = np.zeros_like(label)

        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        for cls in label_class:
            select_pix = np.where(label_tmp == cls)
            if cls in self.list:
                label[select_pix[0], select_pix[1]] = self.list.index(cls) + 1
            else:
                label[select_pix[0], select_pix[1]] = 0

        raw_label = label.copy()

        image, label,padding_mask = self.transform(image, label,padding_mask)

        # Return
        if self.mode == 'val':
            # return image, label, raw_label
            return image, label,padding_mask
        else:
            return image, label,padding_mask