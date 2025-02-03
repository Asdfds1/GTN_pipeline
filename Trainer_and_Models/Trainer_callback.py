import os
import torch
import numpy as np
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import json

class TrainerCallback:
    def __init__(self, log_dir = '../CircuitGen_AI/Run_result', patience=10, use_valid=True):
        self.log_dir = log_dir
        self.patience = patience
        self.best_metric = float('inf') # Инициализируем худшим значением метрики
        self.best_epoch = 0
        self.epochs_since_improvement = 0
        self.best_model_state = None
        self.writer = None
        self.run_dir = None
        self.log_file = None
        self.start_time = None

        self.is_loss = None
        self.use_valid = None

    def set_use_valid(self, use_valid: bool):
        self.use_valid = use_valid
        self.is_loss = not use_valid

    def start_run(self, model_name, metric_name):
        """
        Создает папку для запуска, SummaryWriter и файл для логов.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"{timestamp}_{model_name}_{metric_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_file = open(os.path.join(self.run_dir, "log.txt"), "w")
        self.start_time = time.time()
        self.log(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_epoch_begin(self, epoch):
        """
        Вызывается в начале эпохи
        """
        self.log(f"Epoch {epoch} started")

    def on_epoch_end(self, epoch, model, loss, metric, optimizer, train_time):
        """
        Вызывается в конце каждой эпохи. Логирует, сохраняет чекпоинт лучшей модели, проверяет раннюю остановку.
        """
        # Записываем значения в TensorBoard
        self.writer.add_scalar('loss/train', loss, epoch)
        if self.use_valid:
            self.writer.add_scalar('metric/valid', metric, epoch)
        if train_time:
            self.writer.add_scalar('time/train', train_time, epoch)
        # Логируем в файл
        log_str = f"Epoch: {epoch}, Loss: {loss:.4f}"
        if self.use_valid:
            log_str += f", Metric: {metric:.4f}"
        if train_time:
            log_str += f", Train Time: {train_time:.2f} seconds"
        self.log(log_str)

        # Проверка на улучшение метрики
        if self.is_loss:
            is_best = loss < self.best_metric
        else:
            is_best = metric < self.best_metric

        if is_best:
            self.best_metric = loss if self.is_loss else metric
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
            self.best_model_state = copy.deepcopy(model.state_dict())  # Копируем состояние модели
            self.save_checkpoint(epoch, model, self.best_metric, optimizer)
            self.log("Checkpoint saved, new best model")
        else:
            self.epochs_since_improvement += 1

        # Проверка на раннюю остановку
        if self.epochs_since_improvement >= self.patience and self.use_valid:
            self.log(f"Early stopping at epoch {epoch}")
            self.end_run()
            return True
        return False

    def save_checkpoint(self, epoch, model, metric, optimizer):
        """
        Сохраняет чекпоинт модели.
        """
        checkpoint_path = os.path.join(self.run_dir, "best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric': metric,
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    def load_checkpoint(self, model):
        """
        Загружает чекпоинт модели
        """
        checkpoint_path = os.path.join(self.run_dir, "best_model.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def log(self, message):
        """
        Логирует сообщение в файл.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp}: {message}\n"
        print(log_message, end='')
        self.log_file.write(log_message)
        self.log_file.flush()

    def end_run(self):
        """
        Закрывает файлы и сохраняет конфигурацию обучения.
        :return нужен для загрузки лучшей модели
        """
        self.log(f"Training finished. Best metric {self.best_metric:.4f} at epoch {self.best_epoch}")
        total_time = time.time() - self.start_time
        self.log(f"Total training time: {total_time:.2f} seconds")
        if self.writer:
            self.writer.close()
        if self.log_file:
            self.log_file.close()
        return self.run_dir

    def save_config(self, config):
        """
        Сохраняет конфигурацию обучения в JSON файл
        """
        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        self.log("Config saved")