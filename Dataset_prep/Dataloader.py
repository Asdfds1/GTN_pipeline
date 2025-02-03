import os
import re
import torch
import multiprocessing
import encoders
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from Dataset_prep import typical_error_fixes, Parsers
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class AbstractDataLoader(ABC):
    """
    Абстрактный базовый класс для всех DataLoader-ов.
    """
    @abstractmethod
    def __init__(self, dataset_path, batch_size, parser: Parsers.AbstractDataParser, encoder='One-hot-encoder', shuffle=True, num_workers=1, use_scaler=False,**kwargs):
        """
        Инициализация DataLoader-а.
        Args:
            dataset_path (str): Путь к папке с датасетом.
            batch_size (int): Размер батча.
            shuffle (bool, optional): Перемешивать данные или нет.
            num_workers (int, optional): Количество используемых процессоров для загрузки данных.
            **kwargs: Дополнительные аргументы.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.encoder = encoder
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.parser = parser
        self.kwargs = kwargs
        self.error = self.check_errors()
        if use_scaler:
            self.scaler = self.find_or_fit_scaler()
        else:
            self.scaler =None
            print("Dataloader without Scaler")
        if self.error == 0:
            self.dataset = self.create_dataset() # создаем dataset
        else:
            print('interrupting dataset creation')


    @abstractmethod
    def check_errors(self, path = None) -> int:
        """
        Проверяет ошибки и запускает fixer для типовых ошибок в файлах датасета

        Returns:
            код int, 0 если ошибка исправлена, 1 если нет
        """
        pass

    @abstractmethod
    def create_dataset(self) -> Dataset:
        """Создает и возвращает Dataset"""
        pass

    def find_or_fit_scaler(self) -> MinMaxScaler:
        try:
            with open(self.dataset_path+'_mMscaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                print(f"MinMaxScaler успешно загружен")
                return scaler
        except BaseException as e:
            scaler = MinMaxScaler()
            scaler_folder_data = [(os.path.join(self.dataset_path, f'{folder_name}/{folder_name}AbcStats.json'))
                                  for folder_name in os.listdir(self.dataset_path)
                                  if os.path.isdir(os.path.join(self.dataset_path, folder_name))]

            with multiprocessing.Pool(processes=self.num_workers) as pool:
                labels = list(tqdm(pool.starmap(self.parser.load_label, [(path,) for path in scaler_folder_data]),
                                   total=len(scaler_folder_data),
                                   desc="Fit scaler"))

            # Фильтруем результаты, удаляя None значения, если process_graph_folder может возвращать None
            label = [label for label in labels if label is not None]
            all_labels = np.array([label.item() for label in labels]).reshape(-1, 1)
            scaler.fit(all_labels)
            with open(self.dataset_path+'_mMscaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                print(f"MinMaxScaler успешно сохранен")
                return scaler

    def get_val_train_dataloaders(self, val_size) -> tuple[DataLoader, DataLoader]:
        """
        Создает и возвращает DataLoader для тренировочной и валидационной выборок.

        Returns:
            tuple[DataLoader, DataLoader]: Кортеж из DataLoader для тренировочной и валидационной выборок.
        """
        dataset = self.dataset
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_size)
        train_size = dataset_size - val_size

        # Split dataset in to train and validation datasets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader

    def get_dataloader(self) -> DataLoader:
        """
        Создает и возвращает DataLoader.

        Returns:
            DataLoader: Объект DataLoader.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader

class GraphMLDataLoader(AbstractDataLoader):
    def __init__(self, dataset_path, batch_size, parser, encoder, shuffle=True, num_workers=0, use_scaler=False,**kwargs):
        super().__init__(dataset_path, batch_size, parser, encoder, shuffle, num_workers, use_scaler, **kwargs)

    def check_errors(self, graphml_file_path = None) -> int:
        """
        Проверяет ошибки и запускает fixer для типовых ошибок в файлах датасета

        Returns:
            код int, 0 если ошибка исправлена, 1 если нет
        """
        if graphml_file_path is None:
            graphs = os.listdir(self.dataset_path)
            graphml_file_path = os.path.join(self.dataset_path, f'{graphs[0]}/{graphs[0]}.graphml')

        fix_result = 0

        try:
            with open(graphml_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith("<source="):
                        print(f"Detected edge error in {graphml_file_path}, attempting to fix...")
                        fix_result += typical_error_fixes.FixGraphmlEdgeError(self.dataset_path,
                                                                              self.num_workers).process()
                        break
        except Exception as e:
            print(f"Error during edge fix check: {e} in {graphml_file_path}")
            return 1
        try:
            with open(graphml_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith('<node id="'):
                        parts = stripped_line.split('"')
                        if len(parts) >= 3:
                            node_id = parts[1]
                            if not re.match(r'^[0-9]+$', node_id):
                                print(f"Detected invalid node id format in {graphml_file_path}, attempting to fix...")
                                fix_result += typical_error_fixes.FixGraphmlNodeIdError(self.dataset_path,
                                                                                        self.num_workers).process()
                                break
        except Exception as e:
            print(f"Error during Value fix check: {e} in {graphml_file_path}")
            return 1
        try:
            fix_result += typical_error_fixes.FixInvalidLabelError(self.dataset_path, self.parser).process()
        finally:
            if fix_result > 0:
                print(f'Untypical error in dataset files, pleas check traces')
            else:
                return fix_result

    def process_graph_folder(self, graph_path, graph_name):
        """Обрабатывает папку графом и возвращает Data"""
        graphml_file_path = os.path.join(graph_path, f'{graph_name}.graphml')
        json_file_path = os.path.join(graph_path, f'{graph_name}AbcStats.json')
        one_hot_encoder, edge_index = None, None

        graph = self.parser.load_data(graphml_file_path)
        label = self.parser.load_label(json_file_path)

        if self.scaler:
            label = np.array([label.item()]).reshape(-1, 1)
            label = self.scaler.transform(label)
            label = torch.tensor(label, dtype=torch.long)

        edge_index = torch.tensor(self.get_int_list(graph.edges)).t().long()

        # Извлечение фичей узлов
        node_features = [list(data.values())[0] if data else [0.0] for _, data in graph.nodes(data=True)]

        all_node_types_example = ["and", "buf", "input", "nand", "nor", "not", "or", "output", "xnor", "xor"]

        match self.encoder:
            case 'One-hot-encoder': one_hot_encoder = encoders.OneHotEncoder(all_node_types_example)

        encoded_node_features = encoders.encode_node_features(node_features, one_hot_encoder)

        x = torch.tensor(encoded_node_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=label)
        return data

    def create_dataset(self) -> list:
        """Создает и возвращает Dataset"""

        # Создаем список кортежей (folder_path, graph_name, encoder) для каждой папки
        folder_data = [(os.path.join(self.dataset_path, folder_name), folder_name)
                       for folder_name in os.listdir(self.dataset_path)
                       if os.path.isdir(os.path.join(self.dataset_path, folder_name))]

        # Используем pool.starmap для параллельной обработки, так как process_graph_folder принимает 3 аргумента
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.starmap(self.process_graph_folder, folder_data),
                                total = len(folder_data),
                                desc="Processing graphs"))

        # Фильтруем результаты, удаляя None значения, если process_graph_folder может возвращать None
        graphs_data = [data for data in results if data is not None]

        return graphs_data



    @staticmethod
    def get_int_list(rows):
        result = []
        for row in rows:
            if isinstance(row, tuple):
                if all(map(lambda s: isinstance(s, str) and s.isdigit(), row)):
                    result.append(tuple(map(int, row)))
                else:
                    result.extend(GraphMLDataLoader.get_int_list(row))
        return list(result)