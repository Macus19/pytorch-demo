import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 配置代码运行所用的设备
# 如果有cuda就使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
input_size = 784
hidden_size = 500
num_classes = 10
num_epoch = 5
batch_size = 100
learning_rate = 0.001

# 加载训练集和测试集
train_dataset = torchvision.datasets.MNIST(root="../../data/minist", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="../../data/minist", train=False, transform=transforms.ToTensor())

# 数据加载器，用来读取数据以及进行分批和乱序操作
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

# 构建一个有隐藏层的前馈神经网络类
class NeuralNet(nn.Module):
    # 初始化构造函数
    def __init__(self, input_size, hidden_size, num_classes):
        # 继承父类的构造函数
        super(NeuralNet, self).__init__()
        # 第一个全连接层，输入维度为[batch_size,input_size]，输出维度为[batch_size,hidden_size]
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 使用relu激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层，输入维度为[batch_size,hidden_size]，输出维度为[batch_size,num_classes]
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    # 前馈神经网络运算过程
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定义一个馈神经网络
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器，使用Adam算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

# 训练
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # 转换图片维度
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # 预测
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)

        # 清空梯度缓存
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step: [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epoch, i+1, total_step, loss.item()))

# 测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the 10000 test images {} %".format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), "forward_network_model.ckpt")