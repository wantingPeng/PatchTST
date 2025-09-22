from data_provider.data_factory import data_provider
from data_provider.data_loader_anomaly import get_anomaly_loader
from exp.exp_basic import Exp_Basic
from models import PatchTST, Autoformer, Transformer, Informer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Anomaly(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly, self).__init__(args)
        self.args.anormly_ratio = args.anormly_ratio  # 异常比例，用于计算阈值
        
    def _build_model(self):
        # 根据用户指定的模型类型构建模型
        model_dict = {
            'PatchTST': PatchTST,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        
        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not found. Available models: {list(model_dict.keys())}")
            
        print(f"构建 {self.args.model} 模型...")
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 使用新的数据加载器
        data_set, data_loader = get_anomaly_loader(
            data_path=self.args.data_path,
            batch_size=self.args.batch_size,
            win_size=self.args.win_size,
            step=self.args.step,
            mode=flag,
            num_workers=self.args.num_workers
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')  # 不要立即计算平均值
        return criterion

    def _process_one_batch(self, batch_x):
        batch_x = batch_x.float().to(self.device)
        
        # 根据不同模型类型处理输入
        if 'Linear' in self.args.model or 'TST' in self.args.model:
            # 线性模型和PatchTST模型只需要batch_x
            outputs = self.model(batch_x)
        else:
            # Transformer类模型需要更多输入
            # 对于Autoformer和Transformer模型
            # 创建decoder输入序列
            batch_y = torch.zeros_like(batch_x).float().to(self.device)
            dec_inp = torch.zeros((batch_x.shape[0], self.args.pred_len, batch_x.shape[2])).float().to(self.device)
            if self.args.label_len > 0:
                dec_inp = torch.cat([batch_y[:, -self.args.label_len:, :], dec_inp], dim=1).to(self.device)
            
            # 创建时间特征标记（简化处理，全零）
            time_features_dim = 4  # 时间特征维度
            batch_x_mark = torch.zeros((batch_x.shape[0], batch_x.shape[1], time_features_dim)).float().to(self.device)
            # 为decoder输入创建时间特征标记
            dec_len = self.args.pred_len
            if self.args.label_len > 0:
                dec_len += self.args.label_len
            batch_y_mark = torch.zeros((batch_x.shape[0], dec_len, time_features_dim)).float().to(self.device)
            
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # 只打印一次调试信息
        if hasattr(self, '_printed_debug') == False:
            print(f"输入形状: {batch_x.shape}, 输出形状: {outputs.shape}")
            self._printed_debug = True
        
        # 检查输出与输入的尺寸是否匹配
        if outputs.shape[1] != batch_x.shape[1]:
            print(f"警告：输出时间步长 {outputs.shape[1]} 与输入时间步长 {batch_x.shape[1]} 不匹配")
        
        return outputs, batch_x
    
    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                outputs, batch_x = self._process_one_batch(batch_x)
                
                # 计算重建误差
                loss = self.criterion(outputs, batch_x)
                
                # 按特征和时间步骤计算平均值，这样每个样本有一个标量损失
                loss = torch.mean(loss, dim=(1, 2))  
                
                total_loss.append(loss.detach().cpu())
        
        total_loss = torch.cat(total_loss).numpy()
        return np.mean(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        self.criterion = criterion

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                           steps_per_epoch=train_steps,
                                           pct_start=self.args.pct_start,
                                           epochs=self.args.train_epochs,
                                           max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                outputs, batch_x = self._process_one_batch(batch_x)
                
                # 计算重建误差
                loss = criterion(outputs, batch_x)
                loss = torch.mean(loss)  # 整个批次的平均损失
                
                train_loss.append(loss.item())
                
                # 反向传播和优化
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            test_loss = self.vali(test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                
            # 每个epoch结束保存一次模型，不依赖early stopping
            epoch_path = os.path.join(path, f'epoch_{epoch+1}.pth')
            torch.save(self.model.state_dict(), epoch_path)
            print(f"已保存模型检查点到 {epoch_path}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # 加载训练和测试数据
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        folder_path = './anomaly_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = nn.MSELoss(reduction='none')
        
        # 1. 计算训练集的重建误差，用于确定阈值
        train_scores = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                outputs, batch_x = self._process_one_batch(batch_x)
                # 计算每个样本的重建误差
                score = torch.mean(criterion(outputs, batch_x), dim=(1, 2))
                train_scores.append(score.detach().cpu().numpy())
        
        train_scores = np.concatenate(train_scores, axis=0)
        
        # 2. 计算测试集的重建误差和收集标签 - 逐点评估
        point_scores = []
        point_labels = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                outputs, batch_x = self._process_one_batch(batch_x)
                # 计算每个时间点的重建误差
                errors = criterion(outputs, batch_x)  # [batch_size, seq_len, n_features]
                
                # 对特征维度取平均，保留时间点维度
                point_error = torch.mean(errors, dim=2)  # [batch_size, seq_len]
                
                # 收集每个时间点的标签和分数
                for b in range(batch_y.shape[0]):  # 遍历每个样本
                    point_scores.append(point_error[b].detach().cpu().numpy())
                    point_labels.append(batch_y[b].numpy())
                
                # 保存一些重建结果进行可视化
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    output_np = outputs.detach().cpu().numpy()
                    # 获取当前批次的标签
                    current_labels = batch_y[0].numpy()  # 第一个样本的所有时间点标签
                    has_anomaly = np.any(current_labels == 1)
                    
                    # 选择第一个样本的第一个特征进行可视化
                    plt.figure(figsize=(10, 5))
                    plt.plot(input_np[0, :, 0], label='Input')
                    plt.plot(output_np[0, :, 0], label='Reconstruction')
                    plt.legend()
                    plt.title(f'Sample {i}, Contains Anomaly: {has_anomaly}')
                    plt.savefig(os.path.join(folder_path, f'recon_sample_{i}.png'))
                    plt.close()
        
        # 将所有批次的点级结果合并
        point_scores = np.concatenate(point_scores, axis=0)
        point_labels = np.concatenate(point_labels, axis=0)
        
        # 3. 根据异常率确定阈值
        # 根据经验值确定阈值，例如使用训练数据得分的95%分位数
        threshold = np.percentile(train_scores, 100 - self.args.anormly_ratio)
        print(f"Threshold based on {100 - self.args.anormly_ratio}% percentile: {threshold}")
        
        # 4. 基于阈值生成逐点预测
        point_predictions = (point_scores > threshold).astype(int)
        
        # 5. 计算逐点评估指标
        accuracy = accuracy_score(point_labels, point_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(point_labels, point_predictions, average='binary')
        try:
            auc = roc_auc_score(point_labels, point_scores)
        except:
            auc = 0
        
        print("逐点预测结果评估指标:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # 6. 应用逐点异常检测调整函数
        print("\n应用异常检测调整函数...")
        adjusted_pred = point_predictions.copy()
        gt = point_labels
        
        # 改进的异常检测调整算法 - 限制连续异常的最大范围
        max_adjustment_range = 20  # 限制向前/向后调整的最大范围
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                # 向前调整，但限制最大范围
                for j in range(i, max(0, i-max_adjustment_range), -1):
                    if gt[j] == 0:
                        break
                    else:
                        if adjusted_pred[j] == 0:
                            adjusted_pred[j] = 1
                # 向后调整，但限制最大范围
                for j in range(i, min(len(gt), i+max_adjustment_range)):
                    if gt[j] == 0:
                        break
                    else:
                        if adjusted_pred[j] == 0:
                            adjusted_pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                adjusted_pred[i] = 1
        
        # 7. 计算调整后的逐点评估指标
        adj_accuracy = accuracy_score(point_labels, adjusted_pred)
        adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(point_labels, adjusted_pred, average='binary')
        
        print("\n调整后预测结果评估指标:")
        print(f"Adjusted Accuracy: {adj_accuracy:.4f}")
        print(f"Adjusted Precision: {adj_precision:.4f}")
        print(f"Adjusted Recall: {adj_recall:.4f}")
        print(f"Adjusted F1 Score: {adj_f1:.4f}")
        print(f"AUC: {auc:.4f} (不变)")
        
        # 保存结果
        result_path = os.path.join(folder_path, 'metrics.txt')
        with open(result_path, 'w') as f:
            f.write(f"Threshold: {threshold}\n")
            f.write("\n逐点预测结果评估指标:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            
            f.write("\n调整后逐点预测结果评估指标:\n")
            f.write(f"Adjusted Accuracy: {adj_accuracy:.4f}\n")
            f.write(f"Adjusted Precision: {adj_precision:.4f}\n")
            f.write(f"Adjusted Recall: {adj_recall:.4f}\n")
            f.write(f"Adjusted F1 Score: {adj_f1:.4f}\n")
        
        # 保存异常分数和预测结果
        np.save(os.path.join(folder_path, 'point_scores.npy'), point_scores)
        np.save(os.path.join(folder_path, 'point_labels.npy'), point_labels)
        np.save(os.path.join(folder_path, 'point_predictions.npy'), point_predictions)
        np.save(os.path.join(folder_path, 'adjusted_predictions.npy'), adjusted_pred)
        
        # 保存异常点的分布情况统计
        anomaly_ratio = np.mean(point_labels)
        print(f"\n数据集异常点比例: {anomaly_ratio:.4f} ({np.sum(point_labels)} / {len(point_labels)})")
        with open(os.path.join(folder_path, 'anomaly_stats.txt'), 'w') as f:
            f.write(f"总点数: {len(point_labels)}\n")
            f.write(f"异常点数: {np.sum(point_labels)}\n")
            f.write(f"异常比例: {anomaly_ratio:.4f}\n")
        
        # 返回原始指标和调整后的指标
        return accuracy, precision, recall, f1, adj_accuracy, adj_precision, adj_recall, adj_f1
