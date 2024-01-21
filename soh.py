import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

class Battery_3D(Dataset):
    def __init__(self, flag='train', train_mean=None, train_std=None, scaler_type='std'):
        soh = pd.read_csv('data/SOH/soh.csv')
        grouped = soh.groupby('cycle')
        max_len = 0
        valid_len = []
        for name, group in grouped:
            if len(group) > max_len:
                max_len = len(group)
            valid_len.append(len(group))
        result = np.zeros((len(grouped), max_len, soh.shape[1]-1), dtype=soh.dtypes)
        i = 0
        data_y = []
        for name, group in grouped:
            data_y.append(group['soh_value'].values[0])
            group = group.drop(['soh_value'], axis=1)
            result[i, :len(group), :] = group.values
            i += 1

        mask = np.zeros_like(result, dtype=bool)
        for i in range(result.shape[0]):
            mask[i, :valid_len[i]] = True

        if scaler_type == 'maxmin':
            if flag == 'train':
                data_masked = result[mask].reshape(-1, 130).astype(np.float32)
                max_val = np.max(data_masked, axis=0)
                min_val = np.min(data_masked, axis=0)
                self.max_val = max_val
                self.min_val = min_val
            elif flag == 'test':
                assert train_mean is not None
                assert train_std is not None
                self.min_val = train_mean
                self.max_val = train_std
            result = ((result.reshape(-1, 130).astype(np.float32) - self.min_val) / (self.max_val - self.min_val)).reshape(result.shape)
            result[~mask] = 0
        elif scaler_type == 'std':
            if flag == 'train':
                data_masked = result[mask].reshape(-1, 130).astype(np.float32)
                mean = np.mean(data_masked, axis=0)
                std = np.std(data_masked, axis=0)
                self.means = mean
                self.stds = std
            elif flag == 'test':
                assert train_mean is not None
                assert train_std is not None
                self.means = train_mean
                self.stds = train_std
            result = ((result.reshape(-1, 130).astype(np.float32) - self.means) / self.stds).reshape(result.shape)
            result[~mask] = 0

        self.scaler_type = scaler_type
        self.valid_len = valid_len
        self.columns = soh.columns
        self.target = [8]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        border1s = [0, result.shape[0]//10*8, result.shape[0]//10*9]
        border2s = [result.shape[0]//10*8, result.shape[0]//10*9, result.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = result[border1:border2, :, :]
        self.data_y = np.array(data_y[border1:border2]).reshape(-1, 1)
    
    def __len__(self):
        return len(self.data_y)
    
    def __getitem__(self, index):
        return self.data_x[index, :, :].astype(np.float32), self.data_y[index].astype(np.float32)
    
    def get_valid_len(self):
        return self.valid_len
    
    def get_mean_std(self):
        if self.scaler_type == 'maxmin':
            return self.min_val, self.max_val
        else:
            return self.means, self.stds

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Variable_Len_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, kernel_size=3, padding=1, dropout=0.1):
        super(Variable_Len_RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)
    
    def forward(self, x, valid_len):
        # pack the sequence
        x_packed = pack_padded_sequence(x, valid_len, batch_first=True, enforce_sorted=False)
        # pass through LSTM
        packed_output, _ = self.rnn(x_packed)
        # unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # get the last valid output
        idx = (torch.tensor(valid_len) - 1).view(-1, 1).expand(len(valid_len), output.size(2))
        idx = idx.unsqueeze(1).to(x.device)
        output = output.gather(1, idx).squeeze(1)
        output = self.fc(output)
        return output

class CNN_LSTM_CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, kernel_size=3, padding=1, stride=1, dropout=0.1):
        super(CNN_LSTM_CNN, self).__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding, stride=(4,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size = 1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.rnn = nn.LSTM(128, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7,7), padding=1, stride=(2,4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), padding=1, stride=(2,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size = 1, stride=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size = 1),
            nn.ReLU(),
            nn.Flatten(),
            )
        self.fc = nn.Linear(120, output_size)

    def forward(self, x):
        # CNN
        x = x.unsqueeze(1)
        x = self.cnn_extractor(x)
        x = x.reshape(x.shape[0], x.shape[2], -1)
        rnn_output, _ = self.rnn(x)
        output = self.cnn(rnn_output.unsqueeze(1))
        output = self.fc(output)
        return output
    

import matplotlib.pyplot as plt
import os
def train(net, criterion, train_dataloader, device, epochs):
    def init_xavier(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    net.apply(init_xavier)
    net.to(device)
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=0.0001,
                                     weight_decay=0.01)
    for epoch in range(epochs):
        iter_count = 0
        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        train_losses = []
        # 训练开始
        net.train()
        for i, (data_x, data_y) in enumerate(train_dataloader):
            iter_count += 1
            optimizer.zero_grad()
            data_x = data_x.to(device)
            data_y = data_y.float().to(device)

            output, targets = net(data_x), data_y
            Loss = criterion(output, targets)
            train_losses.append(Loss.item())
            print(f'第{i+1}次循环的损失值: {Loss.item()}')

            Loss.backward()
            optimizer.step()

        print("**************epoch: " + str(epoch) + " 已经结束! **************")
        print("epoch: {}, Loss: {}".format(epoch, np.average(train_losses)))
    folder_path = './soh_results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 绘制训练损失和验证损失图并保存
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss at Epoch {}'.format(epoch+1))
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'train_loss_epoch{}.png'.format(epoch+1)))
    plt.close()

def test(net, test_dataloader, criterion, device):
    net.to(device)
    folder_path = './soh_results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    preds = []
    trues = []

    net.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_dataloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs, target = net(batch_x), batch_y

            outputs = outputs.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            Loss = criterion(torch.from_numpy(outputs), torch.from_numpy(target))
            print(f'第{i+1}次循环的损失值: {Loss.item()}')
            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = target  # batch_y.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)
            trues.append(true)
            
            # 画图
            plt.figure()
            plt.ioff()  # 关闭交互模式
            plt.plot(true, label='GroundTruth', linewidth=3)
            plt.plot(pred, label='Prediction', linewidth=2)
            plt.title('soh')  # 添加标题
            plt.legend()
            plt.savefig(os.path.join(folder_path,'pred{}.png'.format(i+1)), bbox_inches='tight')
            plt.close()

    return


train_data = Battery_3D(flag='train', scaler_type='maxmin')
train_loader = DataLoader(
        train_data,
        batch_size=12,
        shuffle=True,
        num_workers=0,
        drop_last=True)
train_mean, train_std = train_data.get_mean_std()
test_data = Battery_3D(flag='test', train_mean=train_mean, train_std=train_std, scaler_type='maxmin')
test_loader = DataLoader(
        test_data,
        batch_size=12,
        shuffle=False,
        num_workers=0,
        drop_last=False)
loss = nn.MSELoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_CNN(130, 1, 256, 4, kernel_size=(5,3), padding=1, stride=(6,3), dropout=0.1)
Epochs = 100

train(model, loss, train_loader, device, Epochs)
test(model, test_loader, loss, device)

