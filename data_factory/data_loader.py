import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import numpy as np
from utils.tools import target_index

class Dataset_Battery(Dataset):
    def __init__(self, data_path, columns, flag='train', size=None,
                 features='M',target='OT', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.target_index = target_index(columns, target)
        self.scale = scale
        self.columns = columns
        self.scaler = None

        self.data_path = data_path
        self.__read_data__()

    def __format_data__(self):
        if isinstance(self.data_path, list):
            df_list = []
            for path in self.data_path:
                df = pd.read_csv(path, usecols=self.columns)
                df_list.append(df)
            df_raw = pd.concat(df_list)
        else:
            df_raw = pd.read_csv(self.data_path, usecols=self.columns)
        return df_raw
    
    def __read_data__(self):
        df_raw = self.__format_data__()
        border1s = [0, df_raw.shape[0]//10*8 - self.seq_len, df_raw.shape[0]//10*9 - self.seq_len]
        border2s = [df_raw.shape[0]//10*8, df_raw.shape[0]//10*9, df_raw.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw[self.target]

        if self.scale:
            if self.set_type == 0: # train
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            else:
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # encoder_input
        seq_y = self.data_y[r_begin:r_end] # decoder_input
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, target_index=None):
        if target_index is None:
            target_index = self.target_index
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        temp = np.zeros((data.shape[0], self.data_x.shape[-1]))
        temp[:, target_index] = data
        return scaler.inverse_transform(temp)[:, target_index]


class Battery_Pred(Dataset):
    def __init__(self, data_path, columns, flag='train', size=None,
                 features='M',target='OT', scale=True,inverse=False):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        self.features = features
        self.target = target
        self.scale = scale
        self.columns = columns
        self.inverse = inverse
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        df_raw = pd.read_csv(self.data_path, usecols=self.columns)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f).inverse_transform(data)