import pickle

# 加载 scaler.pkl 文件
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# 检查 scaler 的类型并打印均值和标准差
if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
    print("均值:", scaler.mean_)
    print("标准差:", scaler.scale_)
else:
    print("这个缩放器不包含均值和标准差信息。")