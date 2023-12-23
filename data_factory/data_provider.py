from torch.utils.data import DataLoader
from data_factory.data_loader import Dataset_Battery, Battery_Pred

def data_provider(args, flag):
    Data = Dataset_Battery
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Battery_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args['batch_size']

    data_set = Data(
        data_path=args['data_path'],
        flag=flag,
        columns=args['columns'],
        size=[args['size'][0], args['size'][1], args['size'][2]],
        features=args['features'],
        target=args['target'],
        scale=args['scale'],
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    # 
    return data_set, data_loader