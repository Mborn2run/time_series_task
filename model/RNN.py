import torch
import torch.nn as nn
from layers.Series_decomp import series_decomp

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, model_name='LSTM', dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.series_decomp = series_decomp(kernel_size = input_dim)
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(input_dim*2, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'RNN':
            self.rnn = nn.RNN(input_dim*2, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_dim*2, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def forward(self, x):
        res, mean = self.series_decomp(x)
        X = torch.cat([res, mean], dim=-1)
        return self.rnn(X)

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, model_name='LSTM', dropout=0.1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'RNN':
            self.rnn = nn.RNN(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, encoder_outputs):
        outputs, hidden = self.rnn(x.unsqueeze(1), hidden[-1].repeat(self.rnn.num_layers, 1, 1)) # 默认encoder的num_layers的最后一层保存所有信息，所以取最后一层，repeat到decoder的层数上。
        prediction = self.fc(outputs.squeeze(1))
        return prediction, hidden

class Decoder_Attention(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers, model_name='LSTM', num_heads=8, dropout=0.1):
        super(Decoder_Attention, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Linear(output_dim, embedding_dim)
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hidden, encoder_outputs):
        if isinstance(hidden, tuple):  # LSTM
            query = hidden[0][-1].unsqueeze(0)
        else:  # RNN, GRU
            query = hidden[-1].unsqueeze(0)
        key = value = encoder_outputs.permute(1, 0, 2)
        attn_output, _ = self.attention(query, key, value)
        context = attn_output.permute(1, 0, 2)
        x = (self.dropout(self.embedding(x))).unsqueeze(1)
        outputs, hidden = self.rnn(x + context, hidden)
        prediction = self.fc(outputs.squeeze(1))
        return prediction, hidden

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
        encoder_outputs, hidden = self.encoder(source)
        x = target[:, 0, :]
        for t in range(1, label_len):
            output, hidden = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t, :] = output
            x = target[:, t, :]
        for t in range(label_len, label_len+pred_len):
            output, hidden = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t, :] = output
            x = output
        outputs[:, 0, :] = target[:, 0, :]
        return outputs

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, model_name='LSTM', dropout=0.1):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.series_decomp = series_decomp(kernel_size = 11)
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, source, target, label_len, pred_len):
        
        outputs, _ = self.rnn(source)
        prediction = self.fc(outputs)
        return prediction
    

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, kernel_size=3, model_name='LSTM', dropout=0.1):
        super(CNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cnn = nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=1)
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, source, target, label_len, pred_len):
        '''
        In sequence data, we usually perform convolution operations on the length (time or sequence length) 
        dimension because we want to capture local features or patterns of the data on the time or sequence 
        dimension.
        '''
        X = source.permute(0, 2, 1)
        X = self.cnn(X)
        X = X.permute(0, 2, 1)
        outputs, _ = self.rnn(X)
        prediction = self.fc(outputs)
        return prediction