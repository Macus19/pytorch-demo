import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 设备配置
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64
num_epochs = 2
learning_rate = 0.01

# 训练数据下载
train_dataset = torchvision.datasets.MNIST(root="../../data/minist/", train=True, transform=transforms.ToTensor(), download=True)
# 测试数据下载
test_dataset = torchvision.datasets.MNIST(root="../../data/minist/", train=False, transform=transforms.ToTensor())

# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# RNN网络
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 隐藏层个数
        self.num_layers = num_layers
        # lstm单元
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层，将隐藏状态做分类获得结果
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # 初始化细胞状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播获得输出
        out, _ = self.lstm(x, (h0, c0))
        # 获得最后一个时刻的细胞的隐状态
        out = self.fc(out[:, -1,:])
        return out

# 实例化模型
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

torch.save(model.state_dict(), './model/rnn_model.ckpt')