from torch.utils.data import DataLoader
from data_factory.data_loader import Dataset_Series, Series_Pred


def data_provider(args, flag, num_workers=0):
    Data = Dataset_Series
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Series_Pred
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
        data_dim=args['data_dim'],
        timeenc=args['timeenc'],
        freq=args['freq'],
        auto_regression=args['auto_regression']
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set, data_loader