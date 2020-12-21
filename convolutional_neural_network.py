import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 指定运行的gpu设备
torch.cuda.set_device(0)
# 如果有cuda，使用cuda运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# 下载训练数据集
train_dataset = torchvision.datasets.MNIST(root="../../data/minist", train=True, transform=transforms.ToTensor(), download=True)
# 下载测试数据集
test_dataset = torchvision.datasets.MNIST(root="../../data/minist", train=False, transform=transforms.ToTensor())

# 训练集和测试集数据加载器，设置批次输入大小以及乱序
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义卷积神经网络，这里有两个卷积层
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 第一个卷积层
        self.layer1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            # 批归一化
            nn.BatchNorm2d(16),
            # 使用ReLU激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二个卷积层
        self.layer2 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # 批归一化
            nn.BatchNorm2d(32),
            # 使用ReLU激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层
        self.fc = nn.Linear(7*7*32, num_classes)
    
    # 前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# 创建模型，并在gpu上运行
model = ConvNet(num_classes).to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器，使用Adam优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将images和labels两个数据都搬到gpu上
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播计算
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
            print("Epoch [{}/{}], Step[{}/{}], loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 切换成评估模式，测试和训练时的操作不一样
model.eval()

# 测试模型
# 不用计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Test accuracy of the model on the 10000 test images: {} %".format(100 * correct/total))

# 保存模型
torch.save(model.state_dict(), "./model/convModel.ckpt")