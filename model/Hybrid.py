import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos

class Hybrid(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_type, embedding_dim, freq, num_layers, output_dim, model_name='LSTM', dropout=0.1):
        super(Hybrid, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.enc_embedding = DataEmbedding_wo_pos(input_dim, embedding_dim, embed_type=embed_type, freq=freq, dropout=dropout)
                                                  
        if model_name == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, label_len, pred_len):
        inputs = self.enc_embedding(x_enc, x_mark_enc)
        outputs, _ = self.rnn(inputs)
        prediction = self.fc(outputs)
        return prediction