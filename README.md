# Hierarchical Relation Learning for Few-Shot Semantic Segmentation in Remote Sensing Images
HRL-Hierarchical Relation Learning for Few-Shot Semantic Segmentation in Remote Sensing Images

## Installation

## Training base-learners 
Option 1: training from scratch

train_base.sh

Option 2: loading the trained models

Put the provided models in the '/initmodel'

https://github.com/XinnHe/HRL/releases/tag/initmodel%2FBaseNet%2Fisaid
https://github.com/XinnHe/HRL/releases/tag/initmodel%2FBaseNet%2Fdlrsd
https://github.com/XinnHe/HRL/releases/tag/initmodel%2FBaseNet%2FLoveDA


## Train
 
 `python python train.py --shot 1 --split 0 --dataset iSAID --backbone vgg ` 


## Test


### Weights of HRL
https://github.com/XinnHe/HRL/releases/tag/weights



## Citation
> @article{he2025hierarchical,
  title={Hierarchical Relation Learning for Few-shot Semantic Segmentation in Remote Sensing Images},
  author={He, Xin and Liu, Yun and Zhou, Yong and Ding, Henghui and Zhao, Jiaqi and Liu, Bing and Jiang, Xudong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  volume={63},
  pages={4410615},
  publisher={IEEE}
}
