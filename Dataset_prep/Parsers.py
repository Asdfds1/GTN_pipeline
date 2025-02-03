import json
import networkx as nx
import torch
from abc import ABC, abstractmethod
from typing import Any
from lxml import etree


class AbstractDataParser(ABC):
    """
    Базовый класс для всех парсеров данных.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
         """
          Конструктор для инициализации параметров парсера.
          Args:
         """
         pass
    @abstractmethod
    def load_data(self, file_path) -> Any:
        """
        Парсит данные из входного формата в нужный формат.
        Args:
            Путь до файла с данными.
        Returns:
            Any: Данные после парсинга.
        """
        pass

    @abstractmethod
    def load_label(self, file_path)-> Any:
        """
        Парсит labels из входного формата в нужный формат.
        Args:
            Путь до файла с labels.
        Returns:
            Any: Данные после парсинга.
        """
        pass

    def __str__(self):
       """Возвращает имя класса"""
       return self.__class__.__name__

class GraphMLParser(AbstractDataParser):
    """
    Парсит данные из формата GraphML.
    """
    def __init__(self):
        pass

    def load_data(self, file_path):
        """
        Парсит данные из входного формата в нужный формат.
        Args:
            Путь до файла с данными.
        Returns:
            Any: Данные после парсинга.
        """
        try:
            parser = etree.XMLParser()
            tree = etree.parse(file_path, parser)
            graph = nx.parse_graphml(etree.tostring(tree))
            return graph
        except FileNotFoundError as e:
            print(f"Error: graphml file not found at {file_path}")
            return e
        except etree.XMLSyntaxError as e:
            print(f"Error loading graph from {file_path}: {e}")
            return e
        except Exception as e:
            print(f"Error loading graph from {file_path}: {e}")
            return e

    def load_label(self, file_path):
        try:
            with open(file_path, 'r') as f:
                label_data = json.load(f)
                if 'abcStatsBalance' in label_data:
                    abc_stats = label_data['abcStatsBalance']
                    delay = abc_stats.get('delay', 0.0)
                    if delay <= 0:
                        raise ValueError(f"Error: Delay value is not more than zero in {file_path}")
                    # Преобразуем area, delay и edge в torch.tensor
                    return torch.tensor(delay, dtype=torch.long)
                else:
                    print(f"Warning: 'abcStatsBalance' key not found in {file_path}")
                    return None
        except json.JSONDecodeError:
            raise ValueError(f"Error: Invalid JSON in {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading label from {file_path}: {e}")
