import os
from abc import ABC, abstractmethod
import networkx as nx
import os
from tqdm import tqdm
import shutil
import multiprocessing
import json
import torch

class AbstractErrorFixer(ABC):
    """
    Абстрактный базовый класс для обработки ошибок.
    """
    @abstractmethod
    def __init__(self, dataset_path: str, num_workers: int):
        """
            Конструктор
        """
        self.dataset_path = dataset_path
        self.num_workers = num_workers

    @abstractmethod
    def fix_error(self, data):
        """
        Исправляет ошибки в данных.
        Args:
            Объект данных, в котором нужно исправить ошибки.
        Returns:
            Объект данных с исправленными ошибками.
        """
        pass

    def process(self):
        """
        Обрабатывает данные.
        Args:
            Объект данных, который нужно обработать
        Returns:
            Объект данных после обработки
        """
        folder_names = [os.path.join(self.dataset_path, f'{folder_name}/{folder_name}.graphml') for folder_name in os.listdir(self.dataset_path)
                        if os.path.isdir(os.path.join(self.dataset_path, folder_name))]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.fix_error, folder_names), total=len(folder_names), desc="Fixing errors"))
        return sum(results)

    def __str__(self):
        """
        Возвращает имя класса
        """
        return self.__class__.__name__

class FixGraphmlEdgeError(AbstractErrorFixer):

    def __init__(self, dataset_path: str, num_workers):
        super().__init__(dataset_path, num_workers)

    def fix_error(self, graphml_file_path):
        """
        Исправляет ошибки в данных.
        Args:
            Объект данных, в котором нужно исправить ошибки.
        Returns:
            Объект данных с исправленными ошибками.
        """
        try:
            with open(graphml_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            corrected_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith("<source="):
                    corrected_line = f"<edge {stripped_line[1:]}\n"
                    corrected_lines.append(corrected_line)
                else:
                    corrected_lines.append(line)

            with open(graphml_file_path, 'w', encoding='utf-8') as f:
                f.writelines(corrected_lines)
            return 0

        except Exception as e:
            print(f"Error fixing graphml file {graphml_file_path}: {e}")
            return 1

    def __str__(self):
        """
        Возвращает имя класса
        """
        return self.__class__.__name__

class FixGraphmlNodeIdError(AbstractErrorFixer):
    """
    Абстрактный базовый класс для обработки ошибок.
    """

    def __init__(self, dataset_path: str, num_workers):
        super().__init__(dataset_path, num_workers)

    def fix_error(self, graphml_file_path):
        """
        Исправляет ошибки в graphml файле, заменяя буквенные префиксы на числовые в ID узлов.
        """
        try:
            graph = nx.read_graphml(graphml_file_path)
            # 2. Получение текущих id
            old_ids = list(graph.nodes())

            # 3. Создание отображения
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

            # 4. Переименование узлов
            nx.relabel_nodes(graph, id_mapping, copy=False)

            # 5. Сохранение GraphML
            nx.write_graphml(graph, graphml_file_path)
            return 0
        except Exception as e:
            print(f"Error during node id fix check: {e} in {graphml_file_path}")
            return 1

    def __str__(self):
        """
        Возвращает имя класса
        """
        return self.__class__.__name__

class FixInvalidLabelError(AbstractErrorFixer):
    """
    Исправляет ошибку с некорректным значением delay в файле json.
    """

    def __init__(self, dataset_path: str, parser, num_workers=4):
        super().__init__(dataset_path, num_workers)
        self.parser = parser

    def _check_and_get_error_path(self, graph_path, graph_name):
        """
        Проверяет и возвращает путь с ошибкой.
        """
        json_file_path = os.path.join(graph_path, f'{graph_name}AbcStats.json')
        try:
            self.parser.load_label(json_file_path)
            return None
        except ValueError as e:
            print(f"Error detected in {graph_path}: {e}")
            return graph_path

    def fix_error(self, folder_path):
        """
        Удаляет папку с ошибкой
        """
        try:
            shutil.rmtree(folder_path)
            print(f"Folder removed: {folder_path}")
            return 0
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}")
            return 1

    def process(self):
        """
        Обрабатывает данные.
        Args:
            Объект данных, который нужно обработать
        Returns:
            Объект данных после обработки
        """
        folder_names = [os.path.join(self.dataset_path, folder_name) for folder_name in os.listdir(self.dataset_path)
                        if os.path.isdir(os.path.join(self.dataset_path, folder_name))]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            error_paths = list(tqdm(pool.starmap(self._check_and_get_error_path,
                                                 [(folder_name, os.path.basename(folder_name)) for folder_name in
                                                  folder_names]), total=len(folder_names),
                                    desc="Checking label errors"))
        error_paths = [path for path in error_paths if path is not None]

        if len(error_paths) > 0:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                results = list(
                    tqdm(pool.imap(self.fix_error, error_paths), total=len(error_paths), desc="Fixing label errors"))
            return sum(results)
        else:
            return 0

    def __str__(self):
        """
        Возвращает имя класса
        """
        return self.__class__.__name__