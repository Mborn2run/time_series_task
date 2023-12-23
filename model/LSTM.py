import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden, cell))
        prediction = self.fc(outputs.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq_ratio(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq_ratio, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, label_len, pred_len, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_dim = target.shape[-1]

        outputs = torch.zeros(batch_size, pred_len, target_dim).to(self.device)
        hidden, cell = self.encoder(source)

        x = target[:, 0, :]

        for t in range(1, pred_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_force_ratio
            x = target[:, t, :] if teacher_force else output

        return outputs
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, label_len, pred_len):
        batch_size = source.shape[0]
        target_dim = target.shape[-1]

        outputs = torch.zeros(batch_size, label_len + pred_len, target_dim).to(self.device)
        hidden, cell = self.encoder(source)

        x = target[:, 0, :]

        for t in range(1, label_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            x = target[:, t, :]

        for t in range(label_len, label_len+pred_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            x = output
        outputs[:, 0, :] = target[:, 0, :]
        return outputs

class lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, source, target, label_len, pred_len):
        outputs, (hidden, cell) = self.lstm(source)
        prediction = self.fc(outputs)
        return prediction
    
# columns = ['电压', '电流', '功率', '温度', '充电容量', '放电容量', '总容量', '充电能量', '放电能量', '总能量']
# seq_len = 24
# pred_len = 48
# input_data = torch.randn(32, seq_len, len(columns))
# target_data = torch.randn(32, pred_len, len(columns))
# encoder = Encoder(input_dim=len(columns), hidden_dim=128, num_layers=2)
# decoder = Decoder(output_dim=len(columns), hidden_dim=128, num_layers=2)
# model = Seq2Seq(encoder, decoder, device='cpu')
# print(model(input_data, target_data).shape)

