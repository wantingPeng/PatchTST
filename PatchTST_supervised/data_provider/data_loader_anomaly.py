import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ALLContactDataset(Dataset):
    def __init__(self, data_path, win_size, step=1, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # 加载训练数据并进行标准化
        train_data = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
        # 分离特征和标签
        self.train_labels = train_data['anomaly_label'].values
        self.train_data = train_data.drop('anomaly_label', axis=1).values
        
        # 标准化
        self.scaler.fit(self.train_data)
        self.train_data = self.scaler.transform(self.train_data)
        
        # 加载验证和测试数据
        if mode == 'val':
            val_data = pd.read_csv(os.path.join(data_path, 'val_processed.csv'))
            self.val_labels = val_data['anomaly_label'].values
            self.val_data = val_data.drop('anomaly_label', axis=1).values
            self.val_data = self.scaler.transform(self.val_data)
        
        elif mode == 'test':
            test_data = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
            self.test_labels = test_data['anomaly_label'].values
            self.test_data = test_data.drop('anomaly_label', axis=1).values
            self.test_data = self.scaler.transform(self.test_data)
    
    def __len__(self):
        """
        根据不同模式返回数据集长度
        """
        if self.mode == "train":
            return (len(self.train_data) - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (len(self.val_data) - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (len(self.test_data) - self.win_size) // self.step + 1
    
    def __getitem__(self, index):
        """
        返回窗口数据和标签
        """
        index = index * self.step
        
        if self.mode == "train":
            seq_x = self.train_data[index:index + self.win_size]
            seq_y = self.train_labels[index:index + self.win_size]
            return np.float32(seq_x), np.float32(seq_y)
            
        elif self.mode == 'val':
            seq_x = self.val_data[index:index + self.win_size]
            seq_y = self.val_labels[index:index + self.win_size]
            return np.float32(seq_x), np.float32(seq_y)
            
        elif self.mode == 'test':
            seq_x = self.test_data[index:index + self.win_size]
            seq_y = self.test_labels[index:index + self.win_size]
            return np.float32(seq_x), np.float32(seq_y)


def get_anomaly_loader(data_path, batch_size, win_size, step=1, mode='train', num_workers=0):
    """
    创建数据加载器
    """
    dataset = ALLContactDataset(
        data_path=data_path, 
        win_size=win_size,
        step=step,
        mode=mode
    )
    
    shuffle = mode == 'train'
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
    
    return dataset, data_loader
