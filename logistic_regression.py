# 导入的包
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 超参数设置
input_size = 784 # 输入的大小（28*28）
num_classes = 10 # 输出分类的个数
num_epochs = 5 # 迭代的次数
batch_size = 100 # 一次输入的数据量
learning_rate = 0.001 # 学习率

# 加载训练集
# 通过torchvision的这个方法从网上下载MINIST数据集
train_dataset = torchvision.datasets.MNIST(root="../../data/minist", train=True, transform=transforms.ToTensor(), download=True)
# 加载测试集
test_dataset = torchvision.datasets.MNIST(root="../../data/minist", train=False, transform=transforms.ToTensor())

# 数据加载器，用来包装训练和测试时所使用的数据，同时设置批次大小和数据乱序
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 构建一个线性模型
model = nn.Linear(input_size, num_classes)
# 损失函数，交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器，使用随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
# 迭代num_epochs次
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像序列转换至大小为(batch_size, input_size)
        images = images.reshape(-1, 28*28)

        # 前向传播，得到输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播及其优化过程
        # 清空梯度缓存,避免影响到下一个batch
        optimizer.zero_grad()
        # 反向传播，计算新的梯度
        loss.backward()
        # 更新梯度
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}'.format(epoch + 1, num_epochs, i+1, total_step, loss.item()))

# 模型测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # 设为28*28列的数组，行数未知，设为-1
        images = images.reshape(-1, 28*28)
        # 预测输出
        outputs = model(images)
        # _代表最大值，predicted代表最大值所在的index，1为输出所在行的最大值
        _, predicted = torch.max(outputs.data, 1)
        # 返回labels的列数
        total += labels.size(0)
        # 计算正确预测的个数
        correct += (predicted == labels).sum()
    print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))

# 保存训练好的模型
torch.save(model.state_dict(), "model.ckpt")
