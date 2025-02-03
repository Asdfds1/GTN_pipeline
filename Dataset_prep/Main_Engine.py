import json
import importlib
import torch
import os

class MainEngine:
    """Главный класс для запуска обучения или предсказания."""

    def __init__(self, config_path: str):
        """
        Инициализация MainEngine.
        Args:
            config_path (str): Путь к конфигурационному файлу JSON.
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.device = torch.device(self.config["device"])
        self.dataloader = self.create_dataloader()
        self.trainer = self.create_trainer()

    def create_dataloader(self) -> None:
        """Создает DataLoader."""
        dataloader_config = self.config["dataloader"]
        class_name = dataloader_config["name"]
        module = importlib.import_module(f'Dataset_prep.Dataloader')
        dataloader_class = getattr(module, class_name)
        self.dataloader = dataloader_class(dataset_path=self.config["dataset_path"], **dataloader_config["params"])

        print(f"DataLoader {self.dataloader} created.")