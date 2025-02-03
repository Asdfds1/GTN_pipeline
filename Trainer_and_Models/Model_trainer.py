import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
from Trainer_and_Models import Models, custom_criterion
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
import numpy as np
from Trainer_and_Models.Trainer_callback import TrainerCallback
import time
from tqdm import tqdm

class GraphTrainer(ABC):
    """
    Базовый класс для всех графовых моделей.
    """

    @abstractmethod
    def __init__(self, model, optimizer, criterion, device):
        """
        Конструктор для инициализации модели, оптимизатора и критерия.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model.__str__()
        self.criterion_name = criterion.__str__()
        self.callback = TrainerCallback()

    @abstractmethod
    def train_step(self, data):
        """
        Шаг обучения на одном батче.
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        """
        Оценка модели на всем датасете.
        """
        pass
    @abstractmethod
    def predict(self, data_loader):
        """
        Предсказание значений на всем датасете.
        """
        pass
    @abstractmethod
    def fit(self, train_loader, num_epochs):
        """
        Метод обучения модели.
        """
        pass
    @abstractmethod
    def load_best_model(self, run_path: str):
        """
        Загружает лучшую модель из сохраненного чекпоинта
        """
        pass

class TransformerTrainer(GraphTrainer):
    def __init__(self, in_channels, hidden_channels, num_heads, learning_rate, device, criterion:custom_criterion.CustomCriterion):
        """
        Инициализирует Transformer GNN модель, оптимизатор и критерий.
        """
        self.run_dir = ''
        model = Models.TransformerGNN(in_channels, hidden_channels, num_heads)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super().__init__(model, optimizer, criterion, device)

    def train_step(self, data):
        """
        Выполняет шаг обучения на одном батче.
        """
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        out = self.model(data.x, data.edge_index, data.batch)
        loss = self.criterion(out.squeeze(), data.y.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data_loader, criterion=None):
        """
        Оценивает модель на всем датасете. Возвращает значение критерия.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                loss = criterion(out.squeeze(), data.y.unsqueeze(1)) if criterion else self.criterion(out, data.y.unsqueeze(1))
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def predict(self, data_loader):
        """
        Предсказывает значения на всем датасете.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                predictions.extend(out.squeeze().cpu().numpy())
        return np.array(predictions)

    def fit(self, train_loader, valid_loader = None,num_epochs = 10):
        """
        Обучает модель в течение заданного количества эпох.
        """

        self.model.to(self.device)

        if valid_loader:
            self.callback.set_use_valid(True)
        else:
            self.callback.set_use_valid(False)

        self.callback.start_run(model_name=self.model_name, metric_name=self.criterion_name)
        config = {
            'model_params': {
                'in_channels': self.model.in_channels,
                'hidden_channels': self.model.hidden_channels,
                'num_heads': self.model.kwargs['num_heads'],
            },
            'optimizer_params': {
                'learning_rate': self.optimizer.defaults['lr'],
            },
            # 'dataset_params': {
            #     ''
            # },
        }
        self.callback.save_config(config)

        for epoch in range(num_epochs):
            self.callback.on_epoch_begin(epoch)
            start_time = time.time()
            total_loss = 0
            for batch_idx, data in enumerate(tqdm(train_loader,
                                                  desc=f"Epoch {epoch + 1}/{num_epochs}",
                                                  position = 0,
                                                  leave=True)):
                loss = self.train_step(data)
                total_loss += loss
            average_loss = total_loss / len(train_loader)
            metric = None
            if valid_loader:
                metric = self.evaluate(valid_loader, criterion=self.criterion)
            train_time = time.time() - start_time
            if self.callback.on_epoch_end(epoch, self.model, average_loss, metric, self.optimizer, train_time):
                break
        self.run_dir = self.callback.end_run()

    def load_best_model(self, run_path = None):
        if run_path:
            self.run_dir = run_path
        checkpoint_best_model = torch.load(f'{self.run_dir}/best_model.pth')
        self.model.load_state_dict(checkpoint_best_model['model_state_dict'])
        self.model.to(self.device)
