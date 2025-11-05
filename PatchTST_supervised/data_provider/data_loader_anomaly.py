import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler



class CustomLoader(object):
    """
    Custom SegLoader that loads from a single Parquet file, sorts by TimeStamp,
    splits into train/val/test by 70/15/15.
    
    可通过修改类内参数控制训练集异常处理策略：
    - downsample=False: 是否通过下采样正常样本，将训练集异常比例调整到20%
    - onlyNormalData=True: 是否仅保留训练集中的正常数据（过滤掉所有异常标签）
    
    注意：验证集和测试集始终保持原始分布，不受这两个参数影响。
    """
    def __init__(self, data_path, seq_len, step, mode="train"):
        self.mode = mode
        self.step = step
        self.seq_len = seq_len
        self.scaler = StandardScaler()

        # 直接使用传入的 Parquet 完整路径
        if not (isinstance(data_path, str) and data_path.endswith('.parquet') and os.path.isfile(data_path)):
            raise FileNotFoundError(f"传入的 --data_path 必须是存在的Parquet文件路径，当前: {data_path}")
        parquet_file = data_path

        # 读取Parquet并按时间排序
        df = pd.read_parquet(parquet_file)
        if 'TimeStamp' not in df.columns:
            raise ValueError("数据中缺少 'TimeStamp' 列")
        df = df.sort_values(by='TimeStamp').reset_index(drop=True)

        # 分割 70/15/15
        total_len = len(df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        # 统计原始比例（基于行级标签）
        def _ratio(d):
            if 'anomaly_label' in d.columns:
                return float(d['anomaly_label'].sum()) / max(len(d), 1)
            return float('nan')

        orig_train_ratio = _ratio(train_df)
        orig_val_ratio = _ratio(val_df)
        orig_test_ratio = _ratio(test_df)
        
        # 控制参数
        downsample = False  # 是否通过下采样调整异常比例到20%
        onlyNormalData = True  # 是否仅保留训练集中的正常数据（标签=0）
        
        # 参数冲突检查
        if downsample and onlyNormalData:
            print("[警告] downsample 和 onlyNormalData 同时为True，将先下采样再过滤异常，最终训练集将只包含正常数据")
        
        # 训练集重采样到 20% 异常占比：通过随机下采样正常样本，保持时间顺序
        if downsample:
            target_ratio = 0.20
            num_anom_cur = int(train_df['anomaly_label'].sum())
            num_norm_cur = int((1 - train_df['anomaly_label']).sum())
            if num_anom_cur > 0:
                max_norm_keep = int(round(num_anom_cur * (1 - target_ratio) / target_ratio))  # e.g., 4x anomalies
                if num_norm_cur > max_norm_keep:
                    anom_df = train_df[train_df['anomaly_label'] == 1]
                    norm_df = train_df[train_df['anomaly_label'] == 0]
                    norm_keep_df = norm_df.sample(n=max_norm_keep, replace=False, random_state=42)
                    train_df = pd.concat([anom_df, norm_keep_df], axis=0, ignore_index=True)
                    # 恢复按时间顺序
                    if 'TimeStamp' in train_df.columns:
                        train_df = train_df.sort_values(by='TimeStamp').reset_index(drop=True)
                    print("按时间排序")
        
        # 仅保留训练集中的正常数据（过滤掉所有异常标签）
        if onlyNormalData:
            if 'anomaly_label' in train_df.columns:
                num_before = len(train_df)
                num_anom_before = int(train_df['anomaly_label'].sum())
                # 只保留标签为0的数据
                train_df = train_df[train_df['anomaly_label'] == 0].copy()
                train_df = train_df.reset_index(drop=True)
                num_after = len(train_df)
                print(f"[onlyNormalData=True] 训练集过滤异常数据: {num_before} 行 -> {num_after} 行 (移除 {num_anom_before} 个异常样本)")
        
        # 删除 TimeStamp 列
        for d in (train_df, val_df, test_df):
            if 'TimeStamp' in d.columns:
                d.drop(columns=['TimeStamp'], inplace=True)

        # 提取标签
        self.val_labels = val_df['anomaly_label'].values if 'anomaly_label' in val_df.columns else np.zeros(len(val_df))
        self.test_labels = test_df['anomaly_label'].values if 'anomaly_label' in test_df.columns else np.zeros(len(test_df))

        # 特征矩阵（去掉标签列）
        def drop_label(d):
            return d.drop(columns=['anomaly_label']) if 'anomaly_label' in d.columns else d

        train_X = drop_label(train_df).values
        val_X = drop_label(val_df).values
        test_X = drop_label(test_df).values

        # NaN 处理
        train_X = np.nan_to_num(train_X)
        val_X = np.nan_to_num(val_X)
        test_X = np.nan_to_num(test_X)

        # 标准化（仅基于训练集）
        self.scaler.fit(train_X)
        self.train = self.scaler.transform(train_X)
        self.val = self.scaler.transform(val_X)
        self.test = self.scaler.transform(test_X)

        # 打印数据形状与异常比例
        print(f"数据加载自: {parquet_file}")
        print(f"训练集形状: {self.train.shape}")
        print(f"验证集形状: {self.val.shape}")
        print(f"测试集形状: {self.test.shape}")
        print(f"特征维度: {self.train.shape[1]}")

        def _final_ratio(labels):
            labels = np.asarray(labels)
            denom = max(labels.shape[0], 1)
            return float(labels.sum()) / denom

        final_train_ratio = _final_ratio(train_df['anomaly_label'].values) if 'anomaly_label' in train_df.columns else float('nan')
        final_val_ratio = _final_ratio(self.val_labels)
        final_test_ratio = _final_ratio(self.test_labels)

        print("异常比例（行级）:")
        print(f"- 训练集: 原始 {orig_train_ratio*100 if not np.isnan(orig_train_ratio) else float('nan'):.2f}% -> 现有 {final_train_ratio*100 if not np.isnan(final_train_ratio) else float('nan'):.2f}%")
        print(f"- 验证集: {final_val_ratio*100:.2f}%")
        print(f"- 测试集: {final_test_ratio*100:.2f}%")

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.seq_len) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.seq_len) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.seq_len) // self.seq_len + 1

    def __getitem__(self, index):
        if self.mode == "train":
            # 训练模式：返回训练数据窗口和虚拟标签(训练时不使用真实标签)
            index = index * self.step
            return np.float32(self.train[index:index + self.seq_len]), \
                   np.float32(np.zeros(self.seq_len))  # 训练时标签设为0
        elif self.mode == 'val':
            # 验证模式：返回验证数据窗口和对应标签
            index = index * self.step
            return np.float32(self.val[index:index + self.seq_len]), \
                   np.float32(self.val_labels[index:index + self.seq_len])
        elif self.mode == 'test':
            # 测试模式：返回测试数据窗口和对应标签（无重叠窗口）
            index = index * self.seq_len
            return np.float32(self.test[index:index + self.seq_len]), \
                   np.float32(self.test_labels[index:index + self.seq_len])
 


def get_anomaly_loader(data_path, batch_size, seq_len, step=1, mode='train', num_workers=0):
    """
    创建数据加载器
    """
    dataset = CustomLoader(
        data_path=data_path, 
        seq_len=seq_len,
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
