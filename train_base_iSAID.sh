#!/bin/sh
PARTITION=Segmentation

    
GPU_ID=0
GPU_num=1
dataset=isaid  #iSAID # iSAID iSAID_1

arch=BaseNet
net=resnet50 #  vgg resnet50 resnet101
variable1=
variable2=

# PORT=1232
for net in resnet50 vgg  #  resnet50 resnet50
do
        for split in 0 1 2   # 0 1 2
        do
                CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_base_iSAID.py \
                --arch=${arch} \
                --split=${split} \
                --backbone=${net} \
                --dataset=${dataset} \

        done

done


#  使用read命令达到类似bat中的pause命令效果
echo 按任意键继续
read -n 1
echo 继续运行