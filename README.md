# Hierarchical Relation Learning for Few-Shot Semantic Segmentation in Remote Sensing Images
HRL-Hierarchical Relation Learning for Few-Shot Semantic Segmentation in Remote Sensing Images


## Training base-learners 
Option 1: training from scratch
Download the pre-trained backbones from [resnet50_v2.pth+vgg16_bn.pth](https://github.com/XinnHe/HRL/releases/tag/initmodel%2Fbackbone)

`train_base.sh`

Option 2: loading the trained models

Put the provided models in the `/initmodel`

https://github.com/XinnHe/HRL/releases/tag/initmodel%2FBaseNet%2Fisaid
https://github.com/XinnHe/HRL/releases/tag/initmodel%2FBaseNet%2Fdlrsd
https://github.com/XinnHe/HRL/releases/tag/initmodel%2FBaseNet%2FLoveDA


## Train
 
 `python train.py --shot 1 --split 0 --dataset iSAID --backbone vgg ` 


## Test

 `python test.py --shot 1 --split 0 --dataset iSAID --backbone vgg ` 

During **testing**, please manually set the path of the weights via the `resume_path` argument, for example:  
resume_path ="./weights/best_dlrsd_res50_split0_1shot.pth"

### Weights of HRL

We provide the weights of our model for direct use and reproduction.  
- [Download Model Weights](https://github.com/XinnHe/HRL/releases/tag/weights)  


## 📌  **NOTE**
If you don’t feel like running HRL yourself, just leave me a message or drop me an email at `xhe@cumt.edu.cn`. I’ll be happy to **share the HRL visualization results** with you, based on **your visualization style** (blue, red, yellow, green, mask overlays, boundary highlighting, ......).

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
