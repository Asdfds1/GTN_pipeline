import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class CustomCriterion(ABC, nn.Module):
    """
    Абстрактный базовый класс для всех пользовательских критериев.
    """
    @abstractmethod
    def forward(self, output, target):
        """
        Вычисляет лосс
        """
        pass

class MAPE_loss(CustomCriterion):
    def __init__(self, epsilon=1e-6):
        """
        Инициализация loss функции mape
        """
        super(MAPE_loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Вычисления loss значения.
        """
        # Ensure no negative values in y_true
        if torch.any(y_true <= 0):
            raise ValueError("MAPE is undefined for values <= 0 in y_true")

        # Calculate the absolute percentage error
        error = torch.abs((y_true - y_pred) / (y_true + self.epsilon))

        # Remove any infinite values and convert to percentage
        error = torch.where(torch.isinf(error), torch.zeros_like(error), error)

        return torch.mean(error) * 100

class MAE_loss(CustomCriterion):
    def __init__(self):
      """
      Инициализация loss функции mae.
      """
      super(MAE_loss, self).__init__()

    def forward(self, y_pred, y_true):
      """
      Вычисление MAE
      """
      return torch.mean(torch.abs(y_pred - y_true))

class MSE_loss(CustomCriterion):
    def __init__(self):
        """
        Инициализация loss функции mse.
        """
        super(MSE_loss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Вычисление MSE
        """
        return torch.mean((y_pred - y_true) ** 2)