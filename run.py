from data_factory.data_provider import data_provider
import pandas as pd
import torch
import numpy as np
from utils.tools import get_features
from exp.exp_main import train, test, predict
from model.Transformer import Seq2Seq_Transfomer
from model.Autoformer import Autoformer
from model.RNN import Encoder, Decoder, Seq2Seq, Decoder_Attention, RNN, CNN_LSTM
from model.AR import AR, MA, ARIMA
from model.TCN import TemporalConvNet
from model.ANN import ANN
import multiprocessing


if __name__ == '__main__':
        multiprocessing.freeze_support()
        # url = ['data/energy_pred/processed.csv']
        # columns = ['Power', 'airTemperature', 'dewTemperature', 'windSpeed', 'month', 'day', 'weekday', 'hour']
        url = ['data/building_project/data.csv']
        columns = ['室外温度','湿球温度','总负荷', 'month','weekday','day','hour','minute','is_holiday']
        target = ['总负荷']
        # url = ['data/paper_dataset/electricity/electricity_processed.csv']
        # columns = ['month','day','weekday','hour'] + [str(i) for i in range(320)] + ['OT']
        # target = ['OT']
        time_line = [next((element for element in columns if 'time' in element or '时间' in element), None)]
        features = get_features(columns, target)
        batch_size = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epoch = 100
        lr = 0.001
        optimizer = 'adamW'
        # criterion = torch.nn.L1Loss(reduction='sum')
        criterion = torch.nn.MSELoss()
        data_dim = {
                'dim' : 2, # 2 for 2D data, 3 for 3D data, etc.
                'data_shape': [-1, 7, 9], # must satisfy dim and count(-1) <= 1, [data_len, feature1, feature2, ...]
        }
        data_dim_check = True if data_dim['dim'] == 2 else (len(data_dim['data_shape']) == data_dim['dim'] and data_dim['data_shape'].count(-1) <= 1)
        date_unprocessed = 'yes' if 'date' in columns else 'no' # 是否自动处理时间序列数据的date列，目前还没做完。
        auto_regression = {'status': True, 'value': 2} #是否采用自回归模型，若为False，则需要传入对应的target在data中的f_dim
        args = {'batch_size': batch_size, 
                'data_path': url, 
                'size': [88*2, 88, 88], # [seq_len, label_len, pred_len]
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
                'time_line': time_line,
                'data_dim': data_dim,
                'date_unprocessed': date_unprocessed,
                'auto_regression': auto_regression,}

        train_dataset, train_dataloader = data_provider(args, flag='train')
        valid_dataset, valid_dataloader = data_provider(args, flag='val')
        test_dataset, test_dataloader = data_provider(args, flag='test')
        # pred_dataset, pred_dataloader = data_provider(args, flag='pred')

        ar = AR(input_dim=len(columns), seq_len=args['size'][0], pred_len=args['size'][-1], model_type='VAR')
        ma = MA(input_dim=len(columns), seq_len=args['size'][0], pred_len=args['size'][-1], kernel_size=21)
        arima = ARIMA(input_dim=len(columns), seq_len=args['size'][0], pred_len=args['size'][-1], kernel_size=5)

        ann = ANN(input_dim=len(columns), hidden_dim=64, output_dim=len(columns))

        tcn = TemporalConvNet(seq_len=args['size'][0], pred_len=args['size'][-1], num_inputs=len(columns), 
                              num_outputs=len(columns), num_channels=[32], kernel_size=len(columns))
        
        encoder = Encoder(input_dim=len(columns), hidden_dim=16, num_layers=2, model_name='RNN', dropout=0.1)
        decoder = Decoder(output_dim=len(columns), hidden_dim=16, num_layers=3, model_name='RNN', dropout=0.1)
        decoder_attention = Decoder_Attention(output_dim=len(columns), embedding_dim = 32, hidden_dim=32, num_layers=2, num_heads=4, model_name='RNN')
        seq2seq = Seq2Seq(encoder, decoder, device = device)
        seq2seq_attention = Seq2Seq(encoder, decoder_attention, device = device)

        rnn = RNN(input_dim=len(columns), hidden_dim=64, num_layers=1, output_dim=len(columns), model_name='LSTM', dropout=0.1)
        cnn_lstm = CNN_LSTM(input_dim=len(columns), hidden_dim=128, num_layers=2, output_dim=len(columns), kernel_size=1, model_name='LSTM', dropout=0.2)

        transformer = Seq2Seq_Transfomer(input_dim=len(columns), output_dim=len(columns), d_model=128, num_encoder_layers = 2, num_decoder_layers = 2, 
                                        batch_first=True, dim_feedforward = 256)
        
        autoformer_configs = {
                'seq_len': args['size'][0],
                'label_len': args['size'][1],
                'pred_len': args['size'][-1],
                'moving_avg': 5,
                'enc_in': len(columns),
                'dec_in': len(columns),
                'd_model': 256,
                'c_out': len(columns),
                'n_heads': 2,
                'd_ff': 256,
                'dropout': 0.1,
                'activation': 'relu',
                'factor': 1,
                'e_layers': 2,
                'd_layers': 2,
                }
        autoformer = Autoformer(autoformer_configs)

        train(rnn, criterion, train_dataloader, valid_dataloader, args)
        test(rnn, test_dataset, test_dataloader, criterion = criterion, args = args)
        # preds = predict(model, pred_dataloader, args, load=True)
        # print(preds.shape)