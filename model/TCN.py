import torch
import torch.nn as nn
from layers.TCN_layer import TemporalBlock
from layers.Embed import DataEmbedding_wo_pos, TokenEmbedding


class TemporalConvNet(nn.Module):
    '''
    input data shape: (batch_size, seq_len, seq_len)
    seq_len: sequence length
    pred_len: sequence prediction length
    num_channels: your TCN channels [your customed size...]
    kernel_size: len(num_features) means every feature is used in convolution
    '''
    def __init__(self, seq_len, pred_len, embed_type, freq, embedding_dim, 
                 num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = seq_len if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.enc_embedding = DataEmbedding_wo_pos(num_inputs, embedding_dim, embed_type, freq=freq, dropout=dropout)
        self.embedding = TokenEmbedding(embedding_dim, num_outputs)
        self.Linear = nn.Linear(num_channels[-1]*num_inputs, pred_len*num_outputs)

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, label_len, pred_len):
        inputs = self.enc_embedding(batch_x, batch_x_mark)
        tcn = self.network(inputs)
        proj = self.embedding(tcn)
        output = self.Linear(proj.reshape(proj.shape[0], -1))
        return output.reshape(-1, pred_len, batch_x.shape[-1])