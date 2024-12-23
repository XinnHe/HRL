from dataset.base_data import Few_Data, Base_Data


class iSAID_1_few_dataset(Few_Data):

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
    val_class = [list(range(1, 16, 3)), list(range(2, 16, 3)), list(range(3, 16, 3))]
    #[[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]]
    data_root = 'E:\\5-code\data\iSAID'
    train_list = 'E:\\5-code\data\iSAID\\train.txt'
    val_list = 'E:\\5-code\data\iSAID\\val.txt'

    def __init__(self, split=0, shot=1, dataset='iSAID', mode='train', ann_type='mask', transform_dict=None):
        super().__init__(split, shot, dataset, mode, ann_type, transform_dict)


class iSAID_1_base_dataset(Base_Data):
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
    val_class = [list(range(1, 16, 3)), list(range(2, 16, 3)), list(range(3, 16, 3))]
    # [[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]]
    data_root = 'E:\\5-code\data\iSAID'
    train_list ='E:\\5-code\data\iSAID\\train.txt'
    val_list ='E:\\5-code\data\iSAID\\val.txt'

    def __init__(self, split=0, shot=1, data_root=None, dataset='iSAID', mode='train', transform_dict=None):
        super().__init__(split,  data_root, dataset, mode, transform_dict)