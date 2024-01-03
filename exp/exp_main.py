import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import time, os
import numpy as np
from tqdm import tqdm
import Ranger
from utils.tools import EarlyStopping, metric, visual, target_index
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)


def _predict(net, batch_x, batch_y, args):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args['size'][-1]:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args['size'][1], :], dec_inp], dim=1).float().to(args['device'])
        # encoder - decoder
        def _run_model():
            outputs = net(batch_x, dec_inp, args['size'][1], args['size'][2])
            if args['output_attention']:
                outputs = outputs[0]
            return outputs

        outputs = _run_model()
        
        f_dim = target_index(args['columns'], args['target'])  # 请替换为你想要的列的索引
        outputs = outputs[:, -args['size'][-1]:, f_dim]
        batch_y = batch_y[:, -args['size'][-1]:, f_dim].to(args['device'])
        return outputs, batch_y

def train(net, criterion, train_dataloader, valid_dataloader, args):
    def init_xavier(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    if args['init']:
        net.apply(init_xavier)
    path = os.path.join(args['checkpoints'])
    if not os.path.exists(path):
        os.makedirs(path)
    print('training on:', args['device'])
    net.to(args['device'])

    if args['optim'] == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=args['lr'],
                                    weight_decay=0.01)
    elif args['optim'] == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=args['lr'],
                                     weight_decay=0.01)
    elif args['optim'] == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=args['lr'],
                                      weight_decay=0.01)
    elif args['optim'] == 'ranger':
        optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=args['lr'],
                           weight_decay=0.01)
    if args['scheduler_type'] == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args['num_epoch'], eta_min=args['lr_min'])

    train_steps = len(train_dataloader)
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True)
    time_now = time.time()
    for epoch in range(args['num_epoch']):
        iter_count = 0
        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        train_losses = []
        # 训练开始
        net.train()
        for i, (encoeder_inp, decoder_inp) in enumerate(tqdm(train_dataloader, desc='训练')):
            iter_count += 1
            optimizer.zero_grad()
            encoeder_inp = encoeder_inp.float().to(args['device'])
            decoder_inp = decoder_inp.float().to(args['device'])

            output, targets = _predict(net, encoeder_inp, decoder_inp, args)
            Loss = criterion(output, targets)
            train_losses.append(Loss.item())
            print(f'第{i+1}次循环的损失值: {Loss.item()}')
            
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, Loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args['num_epoch'] - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            Loss.backward()
            optimizer.step()

        print("**************epoch: " + str(epoch) + " 已经结束! **************")
        print("epoch: {}, Loss: {}".format(epoch, np.average(train_losses)))
        
        # 测试步骤开始
        net.eval()
        eval_loss = []
        with torch.no_grad():
            for encoeder_inp, decoder_inp in valid_dataloader:
                encoeder_inp = encoeder_inp.float().to(args['device'])
                decoder_inp = decoder_inp.float().to(args['device'])
                output, targets = _predict(net, encoeder_inp, decoder_inp, args)

                pred = output.detach().cpu()
                true = targets.detach().cpu()

                Loss = criterion(pred, true)
                eval_loss.append(Loss)

        print("整体验证集上的Loss: {}".format(np.average(eval_loss)))

        # 绘制训练损失和验证损失图并保存
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(len(eval_loss)), eval_loss, label='Validation Loss', color='red')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss at Epoch {}'.format(epoch+1))
        plt.legend()
        plt.savefig(os.path.join(path, 'train_loss_epoch{}.png'.format(epoch+1)))
        plt.close()

        early_stopping(np.average(eval_loss), net, path, args['name'])
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()
        
    return

def test(net, test_dataset, test_dataloader, criterion, args):
    net.load_state_dict(torch.load(os.path.join(args['checkpoints'], f'{args["name"]}.pth')))
    print('testing on:', args['device'])
    net.to(args['device'])
    preds = []
    trues = []
    folder_path = './test_results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    net.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_dataloader):
            batch_x = batch_x.float().to(args['device'])
            batch_y = batch_y.float().to(args['device'])

            outputs, target = _predict(net, batch_x, batch_y, args)

            outputs = outputs.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            Loss = criterion(torch.from_numpy(outputs), torch.from_numpy(target))
            print(f'第{i+1}次循环的损失值: {Loss.item()}')
            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = target  # batch_y.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)
            trues.append(true)
            
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                f_dim = test_dataset.target_index
                gt = test_dataset.inverse_transform(np.concatenate((input[0][:, f_dim], true[0]), axis=0))
                pd = test_dataset.inverse_transform(np.concatenate((input[0][:, f_dim], pred[0]), axis=0))
                if args['time_line'] is None:
                    time = None
                else:
                    time_dim = target_index(args['columns'], args['time_line'])
                    time = test_dataset.inverse_transform(np.concatenate((input[0][:, time_dim], 
                                                                          batch_y[0][-args['size'][-1]:, time_dim]),
                                                                          axis=0), target_index = time_dim)
                if args['features'] == 'M':
                    for j in range(gt.shape[1]):
                        visual(gt[:, j], pd[:, j], os.path.join(folder_path, str(i) + '_' + str(j) + '.pdf'), title=args['target'][j], x=time)
                else:
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'), title=args['target'], x=time)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = './results/' + args['name'] + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    f = open("result.txt", 'a')
    f.write(args['name'] + "  \n")
    f.write('mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return

def predict(net, pred_dataloader, args, load=False):
    if load:
        path = os.path.join(args['checkpoints'])
        best_model_path = path + '/' + f"{args['name']}.pth"
        logging.info(best_model_path)
        net.load_state_dict(torch.load(best_model_path))
    preds = []
    net.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(pred_dataloader):
            batch_x = batch_x.float().to(args['device'])
            batch_y = batch_y.float().to(args['device'])

            outputs = net(batch_x, batch_y, args['size'][1], args['size'][2])[:, -args['size'][-1]:, :]

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred.reshape(pred.shape[-2], pred.shape[-1]))

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = './results/' + args['name'] + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)

    return preds