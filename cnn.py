import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 加载数据函数，这里用模拟数据替代，你需要替换为实际加载数据的代码
def load_data():
    num_samples = 1000
    time_steps = 100
    features = 9
    data = np.random.randn(num_samples, time_steps, features)
    labels = np.random.randint(0, 2, num_samples)
    return data, labels

# 数据预处理
def preprocess_data(data):
    scaler = StandardScaler()
    num_samples, time_steps, num_features = data.shape
    reshaped_data = data.reshape(-1, num_features)
    scaled_data = scaler.fit_transform(reshaped_data)
    return scaled_data.reshape(num_samples, time_steps, num_features)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * self._get_conv_output_size(input_channels), 128)

    def _get_conv_output_size(self, input_channels):
        dummy_input = torch.randn(1, input_channels, 100)
        x = self.pool1(torch.relu(self.conv1(dummy_input)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度以适应 Conv1d
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return x

# 训练 CNN 并提取特征
def train_cnn_and_extract_features(X_train, X_test, y_train, y_test):
    input_channels = X_train.shape[2]
    model = CNN(input_channels)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    X_test_tensor = torch.Tensor(X_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).expand_as(outputs))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_features = model(X_train_tensor).numpy()
        test_features = model(X_test_tensor).numpy()

    return train_features, test_features

# 训练并评估传统机器学习模型
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"随机森林准确率: {accuracy_score(y_test, rf_pred)}")

    # 支持向量机
    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    print(f"支持向量机准确率: {accuracy_score(y_test, svm_pred)}")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print(f"KNN 准确率: {accuracy_score(y_test, knn_pred)}")

if __name__ == "__main__":
    # 加载数据
    data, labels = load_data()
    # 数据预处理
    processed_data = preprocess_data(data)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
    # 训练 CNN 并提取特征
    X_train_features, X_test_features = train_cnn_and_extract_features(X_train, X_test, y_train, y_test)
    # 训练并评估传统机器学习模型
    train_and_evaluate(X_train_features, X_test_features, y_train, y_test)