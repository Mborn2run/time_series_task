import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Series_decomp import series_decomp


class Seq2Seq_Transfomer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, num_encoder_layers = 4, num_decoder_layers = 4, 
                 batch_first=True, dim_feedforward = 2048): # input_dim = output_dim
        super(Seq2Seq_Transfomer, self).__init__()
        self.embedding = DataEmbedding(c_in=input_dim, d_model=d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.predictor = nn.Linear(d_model, output_dim)
        self.series_decomp = series_decomp(kernel_size = 31)

    def forward(self, src, tgt, label_len, pred_len):
        # target mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src_res, src_mean = self.series_decomp(src)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.predictor(out)