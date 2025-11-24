from models import PatchTST, DLinear
from exp.exp_basic import Exp_Basic
from data_provider.data_loader_anomaly import get_anomaly_loader
from utils.tools import adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime
import json

warnings.filterwarnings('ignore')


class EarlyStopping:
    """
    Custom EarlyStopping class for anomaly detection tasks.
    Supports both 'min' mode (for loss) and 'max' mode (for F1 score).
    Saves model checkpoint and configuration when validation metric improves.
    """
    def __init__(self, patience=3, verbose=False, dataset_name='', delta=0, metadata=None, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = np.inf if mode == 'min' else -np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.metadata = metadata if isinstance(metadata, dict) else None
        self.mode = mode  # 'min' for loss, 'max' for F1/accuracy

    def __call__(self, val_metric, model, path):
        # 根据mode决定score的方向
        if self.mode == 'min':
            score = -val_metric  # loss越小越好
        else:
            score = val_metric   # F1越大越好
            
        if self.best_score is None:
            self.best_score = score
            checkpoint_path = self.save_checkpoint(val_metric, model, path)
            return checkpoint_path
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return None
        else:
            self.best_score = score
            checkpoint_path = self.save_checkpoint(val_metric, model, path)
            self.counter = 0
            return checkpoint_path

    def save_checkpoint(self, val_metric, model, path):
        if self.verbose:
            if self.mode == 'min':
                print(f'Validation loss decreased ({self.val_metric_best:.6f} --> {val_metric:.6f}).  Saving model ...')
            else:
                print(f'Validation F1 increased ({self.val_metric_best:.4f} --> {val_metric:.4f}).  Saving model ...')
        
        # 直接使用path，不创建额外的子目录
        os.makedirs(path, exist_ok=True)
        
        # 保存模型到model.pth
        model_path = os.path.join(path, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # 保存运行参数配置到config.json
        if self.metadata is not None:
            try:
                cfg_path = os.path.join(path, "config.json")
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        self.val_metric_best = val_metric
        
        # 返回保存的checkpoint目录路径
        return path


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        # 保存运行配置为字典，用于保存到config.json
        try:
            self.run_config = vars(args)
        except Exception:
            self.run_config = None
        # 存储最新保存的checkpoint目录路径
        self.latest_checkpoint_dir = None

    def _build_model(self):
        """Build model for anomaly detection. Only supports DLinear and PatchTST."""
        model_dict = {
            'DLinear': DLinear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = get_anomaly_loader(self.args.data_path, self.args.batch_size, self.args.seq_len, self.args.step, flag, self.args.num_workers)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """Validate reconstruction loss on validation set."""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                # Forward pass: both DLinear and PatchTST support direct reconstruction
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def evaluate_anomaly_detection(self, train_loader, vali_loader):
        """Evaluate anomaly detection performance on validation set (used for early stopping)."""
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        # (1) Calculate reconstruction errors on training set (for threshold determination)
        train_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                train_energy.append(score.detach().cpu().numpy())
        
        train_energy = np.concatenate(train_energy, axis=0).reshape(-1)
        
        # (2) Calculate reconstruction errors and labels on validation set
        vali_energy = []
        vali_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                vali_energy.append(score.detach().cpu().numpy())
                vali_labels.append(batch_y)
        
        vali_energy = np.concatenate(vali_energy, axis=0).reshape(-1)
        vali_labels = np.concatenate(vali_labels, axis=0).reshape(-1).astype(int)
        
        # (3) Determine threshold based on training set reconstruction errors
        threshold = np.percentile(train_energy, 100 - self.args.anormly_ratio)
        
        # (4) Predict anomalies on validation set
        pred = (vali_energy > threshold).astype(int)
        
        # (5) Apply point adjustment strategy
        gt, pred = adjustment(vali_labels, pred)
        
        # (6) Calculate metrics
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)
        accuracy = accuracy_score(gt, pred)
        
        self.model.train()
        return f_score, accuracy, precision, recall, threshold

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # 使用新的EarlyStopping，mode='Threshold: 0.000000'表示F1越大越好
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, 
                                      dataset_name=setting, metadata=self.run_config, mode='max')

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                
                # Forward pass: both DLinear and PatchTST support direct reconstruction
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            # 在验证集上评估异常检测性能
            vali_f1, vali_acc, vali_precision, vali_recall, vali_threshold = self.evaluate_anomaly_detection(train_loader, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4e} Vali Loss: {3:.4e}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            print("Vali Anomaly Detection | F1: {0:.4f} Precision: {1:.4f} Recall: {2:.4f} Threshold: {3:.12e}".format(
                vali_f1, vali_precision, vali_recall, vali_threshold))
            
            # 使用F1 score进行early stopping（而不是loss）
            checkpoint_dir = early_stopping(vali_f1, self.model, path)
            if checkpoint_dir is not None:
                self.latest_checkpoint_dir = checkpoint_dir
                print(f"Saved checkpoint to: {checkpoint_dir}")
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        # 加载最佳模型
        # if self.latest_checkpoint_dir is not None and os.path.isdir(self.latest_checkpoint_dir):
        #     best_model_path = os.path.join(self.latest_checkpoint_dir, "model.pth")
        #     self.model.load_state_dict(torch.load(best_model_path))
        #     print(f"Loaded best model from: {best_model_path}")  #在测试前把“磁盘上的最佳权重”加载回内存，确保评估用的就是最佳模型

        return self.model

    def test(self, setting, test=0, checkpoint_path=None):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        
        # 确定checkpoint目录
        if checkpoint_path is not None and os.path.isdir(checkpoint_path):
            # 使用传入的checkpoint路径
            ckpt_dir = checkpoint_path
            print(f"Using checkpoint directory: {ckpt_dir}")
        elif self.latest_checkpoint_dir is not None and os.path.isdir(self.latest_checkpoint_dir):
            # 使用训练时保存的最新checkpoint目录
            ckpt_dir = self.latest_checkpoint_dir
            print(f"Using latest checkpoint directory: {ckpt_dir}")
        else:
            # 否则直接中断程序，抛出错误
            raise RuntimeError("未找到checkpoint目录，程序中断。")
        # 如果有checkpoint目录，从中加载model.pth
        if ckpt_dir is not None:
            model_path = os.path.join(ckpt_dir, "model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        attens_energy = []
        # (1) Calculate reconstruction errors on training set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # Reconstruction
                outputs = self.model(batch_x)
                # Compute reconstruction error
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) Calculate reconstruction errors on test set
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # Reconstruction
            outputs = self.model(batch_x)
            # Compute reconstruction error
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anormly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)


        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # 保存测试结果到result.json（如果使用了checkpoint目录结构）
        if ckpt_dir is not None:
            try:
                result_payload = {
                    "threshold": float(threshold),
                    "summary": {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f_score": float(f_score)
                    }
                }
                result_path = os.path.join(ckpt_dir, "result.json")
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_payload, f, ensure_ascii=False, indent=2)
                print(f"Test results saved to: {result_path}")
            except Exception as e:
                print(f"Failed to save test results: {e}")
        
        return
