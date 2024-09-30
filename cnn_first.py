import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from torchvision.ops import MLP
from tqdm import tqdm


class Config():
    data_path = '../data/wind_dataset.csv'
    timestep = 20  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    out_channels = [10, 20, 30]  # 卷积输出通道
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    epochs = 10  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0003  # 学习率
    model_name = 'cnn'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()
# 1.加载时间序列数据
df = pd.read_csv(config.data_path, index_col=0)
# 2.将数据进行标准化
scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['WIND']).reshape(-1, 1))


# 形成训练数据，例如12345789 12-3456789
def split_data(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y
    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:, 0])
        dataY.append(data[index + timestep][0])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, 1)
    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)
    return [x_train, y_train, x_test, y_test]


# 3.获取训练数据   x_train: 170000,30,1   y_train:170000,7,1
x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)
# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           False)
test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          False)


# 7.定义CNN网络
class CNN(nn.Module):
    def __init__(self, feature_size, out_channels, output_size):
        super(CNN, self).__init__()

        # 定义一维卷积层
        self.conv1d_1 = nn.Conv1d(feature_size, out_channels[0], kernel_size=3)
        self.conv1d_2 = nn.Conv1d(out_channels[0], out_channels[1], kernel_size=3)
        self.conv1d_3 = nn.Conv1d(out_channels[1], out_channels[2], kernel_size=3)

        # 定义输出层
        self.fc1 = nn.Linear(out_channels[2] * 14, 128)
        self.fc2 = nn.Linear(128, output_size)

        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # batch_size, feature_size, timestep[32, 1, 20]

        x = self.conv1d_1(x)  # [32, 10, 18]
        x = self.relu(x)

        x = self.conv1d_2(x)  # [32, 20, 16]
        x = self.relu(x)

        x = self.conv1d_3(x)  # [32, 30, 14]
        x = self.relu(x)

        x = x.flatten(start_dim=1)  # [32, 420]

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


model = MLP(config.feature_size, config.timestep, config.hidden_size, config.output_size)  # 定义MLP网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器
# 8.模型训练
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 config.epochs,
                                                                 loss)
    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), Config.save_path)
print('Finished Training')
# 9.绘制结果
plot_size = 200
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b")
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
plt.legend()
plt.show()
y_test_pred = model(x_test_tensor)
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size]), "b")
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
plt.legend()
plt.show()
