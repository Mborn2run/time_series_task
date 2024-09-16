from data_factory.data_provider import data_provider
import pandas as pd
import torch
import numpy as np
from utils.tools import get_features, KLDivLoss
from exp.exp_main import train, test, predict
from model.Transformer import Seq2Seq_Transfomer
from model.Autoformer import Autoformer
from model.RNN import Encoder, Decoder, Seq2Seq, Decoder_Attention, RNN, CNN_LSTM
from model.AR import AR, MA, ARIMA
from model.TCN import TemporalConvNet
from model.ANN import ANN
from model.TimesNet import TimesNet
from model.ResNet import resnet18
from model.PatchTST import PatchTST
from model.TimeMixer import TimeMixer
import multiprocessing
from torch.distributions import kl_divergence


if __name__ == '__main__':
        multiprocessing.freeze_support()
        url = ['data/energy_pred/processed.csv']
        columns = ['date', 'airTemperature', 'dewTemperature', 'windSpeed', 'Power']
        # url = ['data/building_project/building_date.csv']
        # columns = ['date', 'envir_T','wet_bulb_T','cooling_load']
        target = ['Power']
        # url = ['data/paper_dataset/electricity/electricity.csv']
        # columns = ['date'] + [str(i) for i in range(320)] + ['OT']
        # target = ['OT']
        timeenc = 0 # 0 for pandas time encoding, 1 for custom time encoding
        freq = 'h'
        embed_type = 'fixed'
        features = get_features(columns, target)
        batch_size = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epoch = 100
        lr = 0.001
        optimizer = 'adamW'
        # criterion = torch.nn.L1Loss(reduction='sum')
        criterion = torch.nn.MSELoss()
        # criterion = KLDivLoss()
        data_dim = {
                'dim' : 2, # 2 for 2D data, 3 for 3D data, etc.
                'data_shape': [-1, 7, 9], # must satisfy dim and count(-1) <= 1, [data_len, feature1, feature2, ...]
        }
        data_dim_check = True if data_dim['dim'] == 2 else (len(data_dim['data_shape']) == data_dim['dim'] and data_dim['data_shape'].count(-1) <= 1)
        auto_regression = {'status': True, 'value': 2} #是否采用自回归模型，若为False，则需要传入对应的target在data中的f_dim
        use_amp = False # 是否使用混合精度训练
        args = {'batch_size': batch_size, 
                'data_path': url, 
                'size': [24*7, 0, 24*7], # [seq_len, label_len, pred_len]
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
                'patience':num_epoch//10,
                'checkpoints':'./checkpoints',
                'name': 'seq2seq_LSTM',
                'output_attention': False,
                'timeenc': timeenc,
                'freq': freq,
                'embed_type': embed_type,
                'data_dim': data_dim,
                'auto_regression': auto_regression,
                'use_amp': use_amp}

        train_dataset, train_dataloader = data_provider(args, flag='train')
        valid_dataset, valid_dataloader = data_provider(args, flag='val')
        test_dataset, test_dataloader = data_provider(args, flag='test')
        # pred_dataset, pred_dataloader = data_provider(args, flag='pred')

        ar = AR(input_dim=len(columns)-1, seq_len=args['size'][0], pred_len=args['size'][-1], model_type='VAR')
        ma = MA(input_dim=len(columns)-1, seq_len=args['size'][0], pred_len=args['size'][-1], kernel_size=21)
        arima = ARIMA(input_dim=len(columns)-1, seq_len=args['size'][0], pred_len=args['size'][-1], kernel_size=5)

        ann = ANN(input_dim=len(columns)-1, hidden_dim=64, output_dim=len(columns)-1)

        # TCN存在问题：如果pred_len = 1, 则TCN输出全部为0
        tcn = TemporalConvNet(freq = args['freq'], embed_type = args['embed_type'], embedding_dim=32,
                        seq_len=args['size'][0], pred_len=args['size'][-1], num_inputs=len(columns)-1, 
                        num_outputs=len(columns)-1, num_channels=[32, 64, args['size'][-1]], kernel_size=32)

        resnet = resnet18(num_classes = len(target))
        
        encoder = Encoder(input_dim=len(columns), hidden_dim=16, num_layers=2, model_name='RNN', dropout=0.1)
        decoder = Decoder(output_dim=len(columns), hidden_dim=16, num_layers=3, model_name='RNN', dropout=0.1)
        decoder_attention = Decoder_Attention(output_dim=len(columns), embedding_dim = 32, hidden_dim=32, num_layers=2, num_heads=4, model_name='RNN')
        seq2seq_configs = {
                'seq_len': args['size'][0],
                'label_len': args['size'][1],
                'pred_len': args['size'][-1],
                'embed_type': embed_type,
                'freq' : freq,
                'enc_in': len(columns)-1,
                'dec_in': len(columns)-1,
                'd_model': 16,
                'n_layers': 2,
                'dropout': 0.1,
                'output_dim': len(columns)-1,
                'device': device,
        }
        seq2seq = Seq2Seq(encoder, decoder, seq2seq_configs)
        seq2seq_attention = Seq2Seq(encoder, decoder_attention, seq2seq_configs)

        rnn = RNN(input_dim=len(columns)-1, hidden_dim=12,embedding_dim=6,
                  embed_type=args['embed_type'],freq=args['freq'],num_layers=1, 
                  output_dim=len(columns)-1, model_name='LSTM', dropout=0.1)
        
        cnn_lstm = CNN_LSTM(input_dim=len(columns)-1, hidden_dim=128, num_layers=2, output_dim=len(columns)-1, kernel_size=1, model_name='LSTM', dropout=0.2)

        transformer = Seq2Seq_Transfomer(input_dim=len(columns), output_dim=len(columns), d_model=128, num_encoder_layers = 2, num_decoder_layers = 2, 
                                        batch_first=True, dim_feedforward = 256)
        
        autoformer_configs = {
                'seq_len': args['size'][0],
                'label_len': args['size'][1],
                'pred_len': args['size'][-1],
                'moving_avg': 25,
                'enc_in': len(columns)-1,
                'dec_in': len(columns)-1,
                'embed_type': embed_type,
                'freq' : freq,
                'd_model': 512,
                'c_out': len(columns)-1,
                'n_heads': 2,
                'd_ff': 512,
                'dropout': 0.1,
                'activation': 'relu',
                'factor': 3,
                'e_layers': 2,
                'd_layers': 1,
                }
        autoformer = Autoformer(autoformer_configs)

        timesnet_configs = {
                'task_name': 'long_term_forecast',
                'embed_type': embed_type,
                'seq_len': args['size'][0],
                'label_len': args['size'][1],
                'pred_len': args['size'][-1],
                'enc_in': len(columns)-1,
                'freq' : freq,
                'num_kernels': 6, # for inception_V1
                'd_model': 16,
                'c_out': len(columns)-1,
                'd_ff': 32,
                'dropout': 0.1,
                'e_layers': 2,
                'top_k': 5,
        }
        timesnet = TimesNet(timesnet_configs)

        patch_tst_configs = {
                'task_name': 'long_term_forecast',
                'embed_type': embed_type,
                'seq_len': args['size'][0],
                'label_len': args['size'][1],
                'pred_len': args['size'][-1],
                'enc_in': len(columns)-1,
                'dec_in': len(columns)-1,
                'freq' : freq,
                'e_layers': 1,
                'd_layers': 1,
                'c_out': len(columns)-1,
                'n_heads': 2,
                'd_model': 512,
                'd_ff': 512,
                'dropout': 0.1,
                'factor': 3,
                'output_attention': False,
                'activation': 'gelu',
        }
        patch_tst = PatchTST(patch_tst_configs)

        time_mixer_configs = {
                'task_name': 'long_term_forecast',
                'embed_type': embed_type,
                'seq_len': args['size'][0],
                'label_len': 0,
                'pred_len': args['size'][-1],
                'enc_in': len(columns)-1,
                'dec_in': len(columns)-1,
                'c_out': len(columns)-1,
                'd_model': 16,
                'd_ff': 32,
                'e_layers': 3,
                'd_layers': 1,
                'down_sampling_layers': 3,
                'down_sampling_method': 'avg',
                'down_sampling_window': 2,
                'channel_independence': 0,
                'dropout': 0.1,
                'decomp_method': 'moving_avg',
                'moving_avg': 25,
                'freq': freq,
                'use_norm': 1,
        }
        time_mixer = TimeMixer(time_mixer_configs)

        train(rnn, criterion, train_dataloader, valid_dataloader, args)
        test(rnn, test_dataset, test_dataloader, criterion = criterion, args = args)
        # preds = predict(model, pred_dataloader, args, load=True)
        # print(preds.shape)