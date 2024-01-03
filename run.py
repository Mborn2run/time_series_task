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
        # url = ['data/csv_data/PACK9.csv','data/csv_data/PACK11.csv', 'data/csv_data/PACK12.csv']
        #        'data/csv_data/PACK14.csv', 'data/csv_data/PACK15.csv','data/csv_data/PACK16.csv', 
        #        'data/csv_data/PACK17.csv', 'data/csv_data/PACK18.csv', 'data/csv_data/PACK19.csv',
        #        'data/csv_data/PACK22.csv', 'data/csv_data/PACK23.csv', 'data/csv_data/PACK25.csv',
        #        'data/csv_data/PACK38.csv', 'data/csv_data/PACK66.csv', 'data/csv_data/PACK74.csv',]
        url = ['data/csv_data/PACK9_log.csv']
        # url = ['data/eng_pred/processed.csv']
        # columns = ['电流']
        # columns = ['insulatn_resis','vehl_totl_volt','soc','btry_pak_curnt','hist_prb_temp','lwst_prb_temp','volt1','volt2','volt3','volt4','volt5','volt6','volt7',
        # 'volt8','volt9','volt10','volt11','volt12','volt13','volt14','volt15','volt16','volt17','volt18','volt19','volt20','volt21','volt22','volt23',
        # 'volt24','volt25','volt26','volt27','volt28','volt29','volt30','volt31','volt32','volt33','volt34','volt35','volt36','volt37','volt38','volt39',
        # 'volt40','volt41','volt42','volt43','volt44','volt45','volt46','volt47','volt48','volt49','volt50','volt51','volt52','volt53','volt54','volt55',
        # 'volt56','volt57','volt58','volt59','volt60','volt61','volt62','volt63','volt64','volt65','volt66','volt67','volt68','volt69','volt70','volt71',
        # 'volt72','volt73','volt74','volt75','volt76','volt77','volt78','volt79','volt80','volt81','volt82','volt83','volt84','volt85','volt86','volt87',
        # 'volt88','volt89','volt90','volt91','volt92','volt93','volt94','volt95','volt96','volt97','volt98','volt99','volt100','volt101','volt102','volt103',
        # 'volt104','volt105','volt106','volt107','volt108','volt109','volt110','volt111','volt112','volt113','volt114','volt115','volt116','volt117','volt118',
        # 'volt119','volt120','volt121','volt122','volt123','volt124','volt125','volt126','volt127','volt128','volt129','volt130','volt131','volt132','volt133',
        # 'volt134','volt135','volt136','volt137','volt138','volt139','volt140','volt141','volt142','volt143','volt144','volt145','volt146','volt147','volt148',
        # 'volt149','volt150','volt151','volt152','volt153','volt154','volt155','volt156','volt157','volt158','volt159','volt160','volt161','volt162','volt163',
        # 'volt164','volt165','volt166','volt167','volt168','volt169','volt170','volt171','volt172','volt173','volt174','volt175','volt176','volt177',
        # 'volt178','volt179','volt180','volt181','volt182','volt183','volt184','volt185','volt186','volt187','volt188','volt189','volt190','volt191',
        # 'volt192','Vmax','Vmin','time']
        columns = ['循环时间', '电流', '总容量', 'CAN2_Cell_T1_A1(T)', 'CAN2_Cell_U_N1(V)', 'CAN2_BMSSOC', 'dQ', 'dV', 'dQ/dV', 'dV/dQ', 'dVdQ', 'dI', 'dT', 'dR']
        # columns = ['airTemperature', 'dewTemperature', 'windSpeed', 'hour', 'day_of_week', 'month', 'Power'] # columns' element must conform to the order of the data file columns 
        # target = ['vehl_totl_volt']
        target = ['CAN2_Cell_U_N1(V)']
        # target = ['Power']
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