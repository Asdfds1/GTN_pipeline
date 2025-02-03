import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool
from abc import ABC, abstractmethod


class AbstractGraphModel(ABC, nn.Module):
    """
    Абстрактный базовый класс для всех графовых нейронных сетей.
    """

    @abstractmethod
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        """
        Конструктор для инициализации модели.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, x, edge_index, batch):
        """
        Выполняет прямой проход модели.
        """
        pass

    def __str__(self):
        """ Возвращает имя класса"""
        return self.__class__.__name__

class TransformerGNN(AbstractGraphModel):
    """GNN на основе трансформера."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super().__init__(in_channels, hidden_channels, out_channels, num_heads=num_heads)
        self.conv1 = TransformerConv(-1, hidden_channels, heads=num_heads)
        self.conv2 = TransformerConv(-1, hidden_channels, heads=num_heads)
        self.lin = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.lin2 = nn.Linear(hidden_channels * num_heads, 1)

    def forward(self, x, edge_index, batch): # Добавляем аргумент batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.lin(x)
        x = self.lin2(x)
        x = global_mean_pool(x, batch) # Используем глобальный пулинг для агрегации по узлам
        return x


class GCNModel(AbstractGraphModel):
    """
    Модель графовой сверточной нейронной сети (GCN).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
          self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        """
        Прямой проход модели.
        """

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return x

