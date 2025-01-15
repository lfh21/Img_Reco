import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from dataset import Dataset
import matplotlib.pyplot as plt
import os

# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 池化层：最大池化，窗口大小2x2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 100)


    def forward(self, x):
        # 应用卷积层1和激活函数ReLU
        x = F.relu(self.conv1(x))
        # 应用池化层
        x = self.pool(x)
        # 应用卷积层2和激活函数ReLU
        x = F.relu(self.conv2(x))
        # 应用池化层
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # 展平张量
        x = x.view(-1, 32 * 8 * 8)
        # 应用全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    print("Current working directory:", os.getcwd())
    data = Dataset(r"imagenet_mini",
                   dataset='CNN',downsample_rate=0,gray=True)
    data.load()
    data.get_labels()
    train_data, train_labels, test_data, test_labels = data.get_data()
    # 将张量列表转换为一个整体张量
    train_data_tensor = torch.stack(train_data)
    train_labels_tensor = torch.stack(train_labels)

    # 对测试数据进行相同操作
    test_data_tensor = torch.stack(test_data)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)



    # 创建数据加载器，对数据进行打包
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    # 创建模型实例
    model = SimpleCNN().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_values = []
    # 训练模型
    for epoch in range(20):  # 迭代30个epoch
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            # 将梯度缓存清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            # if epoch == 19:
            #     print("final epoch")
            if i % 10 == 9:  # 每200个小批量打印一次
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
        loss_values.append(running_loss / len(train_loader))
            
    print('Finished Training')
    # 保存模型参数
    torch.save(model.state_dict(), 'simple_cnn.pth')
    print('Model parameters saved to simple_cnn.pth')


    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    # with torch.no_grad():
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _,targ = torch.max(labels,1)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == targ).sum().item()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')