'''
LastEditors: Yygz314 2711859393@qq.com
LastEditTime: 2025-04-12 20:39:50
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置中文字体和解决负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体或其他中文字体（如'Microsoft YaHei'）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 设置数据路径
data_dir = r'C:\Users\yan-cheng\Desktop\跌倒检测模型\sisfall\SisFall_dataset_processed'

def load_and_prepare_data():
    # 加载合并后的数据集
    d_data = pd.read_csv(os.path.join(data_dir, 'D_all.csv'))
    f_data = pd.read_csv(os.path.join(data_dir, 'F_all.csv'))
    # 添加标签
    d_data['label'] = 0  # D数据集标记为0
    f_data['label'] = 1  # F数据集标记为1
    # 合并数据集
    all_data = pd.concat([d_data, f_data], ignore_index=True)
    # 分割特征和标签
    X = all_data.drop('label', axis=1).values
    y = all_data['label'].values
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 只需返回2D数组格式的数据即可，不需要reshape为3D
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # 创建随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred)
    
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = ['D类', 'F类']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    # 加载和准备数据
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # 训练和评估模型
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)