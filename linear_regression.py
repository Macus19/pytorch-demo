import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 超参数设置
input_size = 1 # 输入的大小
output_size = 1 # 输入的大小
num_epochs = 60 # 迭代次数
learning_rate = 0.001 # 学习率

# dataset，随机设置的
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 设置线性回归模型
model = nn.Linear(input_size, output_size)

# 损失函数和优化器
# 损失函数为均方损失函数
criterion = nn.MSELoss()
# 优化器，使用随机梯度下降，lr代表学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 对模型进行训练
# 数据集迭代num_epochs次
for epoch in range(num_epochs):
    # numpy数组转换为torch张量
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # 前向传播
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    # 反向传播，计算新的梯度
    loss.backward()
    # 梯度更新
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 绘制图像
# torch.from_numpy将x_train转换为Tensor张量
# model根据输入和模型得到输出
# detach().numpy()将预测结果转换为numpy数组
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label="Original Data")
plt.plot(x_train, predicted, label="Fitted line")
plt.legend()
# 绘制图像
plt.show()

# 将模型的记录节点保存下来
torch.save(model.state_dict(), "model.ckpt")