import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, model_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, model_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + f'{model_name}.pth')
        self.val_loss_min = val_loss

def get_features(columns, target):
    if len(columns) == 1 and len(target) == 1 and columns[0] == target[0]:
        return 'S'
    elif len(columns) > 1 and len(target) == 1 and target[0] in columns:
        return 'MS'
    elif len(columns) > 1 and len(target) > 1 and all(t in columns for t in target):
        return 'M'
    else:
        raise ValueError('输入的columns和target不符合任何一个条件')

def target_index(columns, target):
    indices = [columns.index(t) for t in target if t in columns]
    return indices

def visual(true, preds=None, name='./pic/test.pdf', title='test', x=None):
    """
    Results visualization
    """
    plt.figure()
    plt.ioff()  # 关闭交互模式
    if x is None:
        plt.plot(true, label='GroundTruth', linewidth=3)
        if preds is not None:
            plt.plot(preds, label='Prediction', linewidth=2)
    else:
        plt.plot(x, true, label='GroundTruth', linewidth=3)
        if preds is not None:
            plt.plot(x, preds, label='Prediction', linewidth=2)
    plt.title(title)  # 添加标题
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe