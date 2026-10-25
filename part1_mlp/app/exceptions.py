"""ошибки, которые могут возникнуть в проекте"""


class NeuralNetworkError(Exception):
    """базовая ошибка, связанная с работой нейросети"""


class InvalidLayerSizeError(NeuralNetworkError):
    """ошибка возникает, если указаны неправильные размеры слоев"""


class InvalidActivationError(NeuralNetworkError):
    """ошибка возникает, если указана неизвестная функция активации"""


class InvalidTrainingParamsError(NeuralNetworkError):
    """ошибка возникает, если параметры обучения неравильно заданы"""


class MismatchedDataError(NeuralNetworkError):
    """ошибка возникает, если данные не подходят для работы"""


class DataFileError(NeuralNetworkError):
    """ошибка возникает, если поврежден файл с данными"""


class NetworkNotReadyError(NeuralNetworkError):
    """ошибка возникает, если данные или сеть не готовы к работе"""
