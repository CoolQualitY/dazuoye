import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),               # 调整图像大小
    transforms.RandomHorizontalFlip(),           # 随机水平翻转
    transforms.RandomRotation(15),              # 随机旋转
    transforms.ToTensor(),                      # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 定义数据集
train_dataset = datasets.ImageFolder(root='chest_xray/train', transform=transform)
val_dataset = datasets.ImageFolder(root='chest_xray/val', transform=transform)
test_dataset = datasets.ImageFolder(root='chest_xray/test', transform=transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 输入3通道，输出16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器部分
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, 512)  # 根据图像大小调整
        self.fc2 = nn.Linear(512, 2)  # 2类：肺炎和正常

    def forward(self, x):
        x = self.encoder(x)  # 通过编码器提取特征
        x = self.flatten(x)  # 展平为一维
        x = self.fc1(x)  # 第一层全连接
        x = self.fc2(x)  # 第二层全连接（分类器）
        return x


# 初始化模型
model = AutoencoderCNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train(model, train_loader, criterion, optimizer, scheduler, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 清除之前的梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            running_loss += loss.item()

        scheduler.step()  # 调整学习率
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return train_losses


# 训练模型
train_losses = train(model, train_loader, criterion, optimizer, scheduler, epochs=10)

# 绘制损失曲线
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()


def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 获取预测标签
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


# 评估模型
evaluate(model, test_loader)