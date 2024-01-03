from data_factory.data_provider import data_provider
import pandas as pd
import torch
import numpy as np
from utils.tools import get_features
from exp.exp_main import train, test, predict
from model.Transformer import Seq2Seq_Transfomer
from model.RNN import Encoder, Decoder, Seq2Seq, Decoder_Attention, RNN
import multiprocessing


if __name__ == '__main__':
        multiprocessing.freeze_support()
        url = ['data/eng_pred/processed.csv']
        columns = ['airTemperature', 'dewTemperature', 'windSpeed', 'hour', 'day_of_week', 'month', 'Power'] # columns' element must conform to the order of the data file columns 
        target = ['Power']
        time_line = [next((element for element in columns if 'time' in element or '时间' in element), None)]
        features = get_features(columns, target)
        batch_size = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epoch = 20
        lr = 0.005
        optimizer = 'adamW'
        # criterion = torch.nn.L1Loss(reduction='sum')
        criterion = torch.nn.MSELoss()
        args = {'batch_size': batch_size, 
                'data_path': url, 
                'size': [24, 5, 5], 
                'columns': columns, 
                'features': features, 
                'target': target, 
                'scale':True,
                'device': device,
                'num_epoch': num_epoch,
                'lr': lr,
                'lr_min': 0.00001,
                'optim':optimizer,
                'scheduler_type':'Cosine',
                'init':True,
                'patience':5,
                'checkpoints':'./checkpoints',
                'name': 'seq2seq_LSTM',
                'output_attention': False,
                'time_line': time_line,}

        train_dataset, train_dataloader = data_provider(args, flag='train')
        valid_dataset, valid_dataloader = data_provider(args, flag='val')
        test_dataset, test_dataloader = data_provider(args, flag='test')
        # pred_dataset, pred_dataloader = data_provider(args, flag='pred')

        encoder = Encoder(input_dim=len(columns), hidden_dim=16, num_layers=1, model_name='RNN', dropout=0.1)
        decoder = Decoder(output_dim=len(columns), hidden_dim=16, num_layers=1, model_name='RNN', dropout=0.1)
        decoder_attention = Decoder_Attention(output_dim=len(columns), embedding_dim = 64, hidden_dim=64, num_layers=2, num_heads=4, model_name='LSTM')
        seq2seq = Seq2Seq(encoder, decoder, device = device)
        seq2seq_attention = Seq2Seq(encoder, decoder_attention, device = device)

        rnn = RNN(input_dim=len(columns), hidden_dim=16, num_layers=1, output_dim=len(columns), model_name='RNN', dropout=0.1)

        transformer = Seq2Seq_Transfomer(input_dim=len(columns), output_dim=len(columns), d_model=128, num_encoder_layers = 2, num_decoder_layers = 2, 
                                        batch_first=True, dim_feedforward = 256)

        train(rnn, criterion, train_dataloader, valid_dataloader, args)
        test(rnn, test_dataset, test_dataloader, criterion = criterion, args = args)
        # preds = predict(model, pred_dataloader, args, load=True)
        # print(preds.shape)