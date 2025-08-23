r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL, BaseDatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.isaid import DatasetISAID, BaseDatasetISAID
from data.dlrsd import DatasetDLRSD, BaseDatasetDLRSD


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'isaid': DatasetISAID,
            'dlrsd': DatasetDLRSD,
        }
        cls.basedatasets = {
            'pascal': BaseDatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'isaid': BaseDatasetISAID,
            'dlrsd': BaseDatasetDLRSD,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

        cls.transform_vis = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                                transforms.ToTensor()])
        '''
        cls.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

        '''

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, mode, aug=False, shot=1, classtype='base'):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = mode == 'trn'
        nworker = nworker if mode == 'trn' else 0
        if classtype == 'base':
            dataset = cls.basedatasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, mode=mode,
                                                  use_original_imgsize=cls.use_original_imgsize, aug=aug)
        else:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, mode=mode, shot=shot,
                                              use_original_imgsize=cls.use_original_imgsize, aug=aug)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True,
                                num_workers=nworker)  # ,pin_memory=True,drop_last=True)

        return dataset, dataloader
