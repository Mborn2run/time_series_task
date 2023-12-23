from data_factory.data_provider import data_provider
import pandas as pd
import torch
import numpy as np
from exp.exp_main import train, test, predict
from model.LSTM import Encoder, Decoder, Seq2Seq, Seq2Seq_ratio, lstm

# url = ['data/csv_data/PACK9.csv','data/csv_data/PACK11.csv', 'data/csv_data/PACK12.csv']
#        'data/csv_data/PACK14.csv', 'data/csv_data/PACK15.csv','data/csv_data/PACK16.csv', 
#        'data/csv_data/PACK17.csv', 'data/csv_data/PACK18.csv', 'data/csv_data/PACK19.csv',
#        'data/csv_data/PACK22.csv', 'data/csv_data/PACK23.csv', 'data/csv_data/PACK25.csv',
#        'data/csv_data/PACK38.csv', 'data/csv_data/PACK66.csv', 'data/csv_data/PACK74.csv',]
url = ['data/csv_data/test.csv']
# columns = ['电流', 'CAN2_Cell_T1_A1(T)', 'CAN2_BMSSOC', 'CAN2_Cell_U_N1(V)']
columns = ['Temperature']
batch_size = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epoch = 20
lr = 0.005
optimizer = 'adamW'
# criterion = torch.nn.L1Loss(reduction='sum')
criterion = torch.nn.MSELoss()
args = {'batch_size': batch_size, 
        'data_path': url, 
        'size': [5, 1, 1], 
        'columns': columns, 
        'features': 'S', 
        'target':'Temperature', 
        'scale':True,
        'device': device,
        'num_epoch': num_epoch,
        'lr': lr,
        'lr_min': 0.00001,
        'optim':optimizer,
        'scheduler_type':'Cosine',
        'init':True,
        'patience':7,
        'checkpoints':'./checkpoints',
        'name': 'seq2seq_LSTM',
        'output_attention': False,}

train_dataset, train_dataloader = data_provider(args, flag='train')
valid_dataset, valid_dataloader = data_provider(args, flag='val')
test_dataset, test_dataloader = data_provider(args, flag='test')
# pred_dataset, pred_dataloader = data_provider(args, flag='pred')

encoder = Encoder(input_dim=len(columns), hidden_dim=48, num_layers=4)
decoder = Decoder(output_dim=len(columns), hidden_dim=128, num_layers=4)
model = Seq2Seq(encoder, decoder, device = device)
net = lstm(input_dim=len(columns), hidden_dim=24, num_layers=2, output_dim=len(columns))

# train(net, criterion, train_dataloader, valid_dataloader, args)
test(net, test_dataset, train_dataloader, args)
# preds = predict(model, pred_dataloader, args, load=True)
# print(preds.shape)