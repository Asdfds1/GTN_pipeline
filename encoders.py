from abc import ABC, abstractmethod

class NodeFeatureEncoder(ABC):
    """
    Абстрактный базовый класс для всех энкодеров фичей узлов.
    """
    @abstractmethod
    def encode(self, node_features):
        """
        Кодирует фичи узлов.
        """
    pass

class OneHotEncoder(NodeFeatureEncoder):
    """
    Кодировщик, использующий one-hot encoding.
    """
    def __init__(self, all_node_types):
        """
        Создает словарь для кодирования типов узлов в one-hot вектора.
        """
        self.node_type_to_encoding = {
            node_type: [int(i == idx) for i in range(len(all_node_types))]
            for idx, node_type in enumerate(all_node_types)
        }
    def encode(self, node_features):
        """
        Кодирует фичи узлов с помощью one-hot encoding и словаря.
        """
        encoded_features = []
        for feature in node_features:
            if feature in self.node_type_to_encoding:
                 encoded_features.append(self.node_type_to_encoding[feature])
            else:
                encoded_features.append([0] * len(next(iter(self.node_type_to_encoding.values()))))
        return encoded_features

def encode_node_features(node_features, encoder):
    """
    Кодирует фичи узлов с помощью переданного энкодера
    """
    encoded_features = encoder.encode(node_features)
    return encoded_features