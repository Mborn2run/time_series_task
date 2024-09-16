import torch
import torch.nn as nn


class AR(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, model_type='VAR'):
        super(AR, self).__init__()
        self.linear = nn.Linear(seq_len*input_dim, pred_len*input_dim)
        self.model_type = model_type

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, label_len, pred_len):
        if self.model_type == 'VAR':
            x = batch_x.reshape(batch_x.shape[0], -1)
        output = self.linear(x)
        output = output.view(-1, pred_len, batch_x.shape[-1])
            
        return output


class MA(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, kernel_size=3):
        super(MA, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size-1)//2)  # MA部分
        self.Linear = nn.Linear(seq_len*input_dim, pred_len*input_dim)

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, label_len, pred_len):
        # MA部分
        x = batch_x.reshape(batch_x.shape[0], -1)
        x = x.unsqueeze(1)  # 添加一个维度以适应卷积操作
        mean = self.conv(x)
        res = (x - mean )/torch.std(batch_x)# 残差近似标准化后作为白噪声
        ma = self.Linear(res.squeeze(1)) + torch.mean(batch_x)  # wθ + mean, θ为白噪声
        return ma.reshape(-1, pred_len, batch_x.shape[-1])


class ARIMA(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, kernel_size=3):
        super(ARIMA, self).__init__()
        self.ar_linear = nn.Linear((seq_len - 1)*input_dim, pred_len*input_dim)  # AR部分
        self.ma_conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size-1)//2)  # MA部分
        self.ma_linear = nn.Linear((seq_len - 1)*input_dim, pred_len*input_dim)
        self.diff = nn.Parameter(torch.zeros(pred_len*input_dim)) # 差分部分

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, label_len, pred_len):
        source_diff = batch_x.clone()
        # 差分部分
        for i in range(batch_x.shape[-1]):
            source_diff[:, 1:, i] = batch_x[:, 1:, i] - batch_x[:, :-1, i]*self.diff[i]
        source_diff = source_diff[:, 1:, :]
        x = source_diff.reshape(source_diff.shape[0], -1)

        # AR部分
        ar = self.ar_linear(x)
        
        # MA部分
        x = x.unsqueeze(1) 
        mean = self.ma_conv(x)
        res = (x - mean )/torch.std(source_diff)# 残差近似标准化后作为白噪声
        ma = self.ma_linear(res.squeeze(1)) + torch.mean(source_diff)  # wθ + mean, θ为白噪声
        
        # 将AR部分和MA部分结合起来
        out = ar + ma
        return out.reshape(-1, pred_len, source_diff.shape[-1])