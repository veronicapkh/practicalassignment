"""нейросеть, обучающаяся на данных и делающая предсказания"""
import json
from pathlib import Path
import numpy as np
from app.exceptions import (DataFileError,
                            InvalidActivationError,
                            InvalidLayerSizeError,
                            InvalidTrainingParamsError,
                            MismatchedDataError)
activations = ("sigmoid", "relu")


def sigmoid(values):
    """нейросеть преобразует числа в значения от 0 до 1
    параметры: values: числа для преобразования
    возвращает: числа с новыми значениями"""
    return 1.0/(1.0+np.exp(-np.clip(values, -500, 500)))


def relu(values):
    """relu: отрицательные числа заменяет на ноль
     параметры: values: числа для преобразования
    возвращает: числа с новыми значениями"""
    return np.maximum(0.0, values)


class NeuralNetwork:
    """нейросеть, состоящая из нескольких связанных скрытых слоев
    скрытые слои с выбранной функцией активации
    послений слой всегда выдает значения от 0 до 1"""

    def __init__(self, layer_sizes, activation="sigmoid"):
        """создает нейросеть со случайными весами
        параметры: layer_sizes: список размеров слоев
                   activation: функция активайии "sigmoid" или "relu" """
        if len(layer_sizes) < 3:
            raise InvalidLayerSizeError(
                "нужно минимум 3 слоя: вход, скрытый, выход")
        if any(size < 1 for size in layer_sizes):
            raise InvalidLayerSizeError(
                "размер каждого слоя должен быть больше ноля")
        self.layer_sizes = list(layer_sizes)
        self.set_activation(activation)
        self.weights = []
        self.biases = []
        generator = np.random.default_rng()
        for index in range(len(self.layer_sizes)-1):
            shape = (self.layer_sizes[index], self.layer_sizes[index+1])
            self.weights.append(generator.normal(0, 0.5, shape))
            self.biases.append(np.zeros((1, shape[1])))
        self.loss_history = []

    def set_activation(self, name):
        """выбирает функцию активации для скрытых слоев
        параметры: name: название функции "sigmoid" или "relu" """
        if name not in activations:
            raise InvalidActivationError(
                f"неизвестная функция: {name}. доступно: sigmoid, relu")
        self.activation_name = name

    def _activate(self, values):
        """применяет выбранную функцию активации к значениям"""
        if self.activation_name == "relu":
            return relu(values)
        return sigmoid(values)

    def _derivative(self, layer_output):
        """вычисляет значения, нужные для обучения сети"""
        if self.activation_name == "relu":
            return (layer_output > 0).astype(float)
        return layer_output*(1.0-layer_output)

    def _check_inputs(self, inputs):
        """проверяет входные данные и приводит их к нужному виду"""
        try:
            table = np.asarray(inputs, dtype=float)
        except ValueError as error:
            raise MismatchedDataError(
                f"во входных данных есть не числа: {error}") from error
        if table.ndim == 1:
            table = table.reshape(1, -1)
        if table.ndim != 2 or table.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"сеть ждет {self.layer_sizes[0]} признаков на примере")
        return table

    def forward(self, inputs):
        """передает данные через все слои сети
        параметры: input: входные данные для сети
        возвращает: список результатов работы каждого слоя"""
        layer_output = self._check_inputs(inputs)
        outputs = [layer_output]
        last_index = len(self.weights)-1
        for index in range(len(self.weights)):
            raw = layer_output @ self.weights[index]+self.biases[index]
            if index == last_index:
                layer_output = self._activate(raw)
            else:
                layer_output = self._activate(raw)
            outputs.append(layer_output)
        return outputs

    def predict(self, inputs):
        """получает предсказание для входных данных
        параметры: inputs: один пример или несколько примеров
        возвращает: ответы нейросети для переданных примеров"""
        return self.forward(inputs)[-1]

    def backward(self, outputs, targets, learning_rate):
        """исправляет веса сети на основе допущенных ошибок
        параметры: outputs: результаты работы слоев сети
                targets: правильные ответы
                learning_rate: скорость изменения весов"""
        batch_size = targets.shape[0]
        final = outputs[-1]
        delta = (final-targets)*final*(1.0-final)
        for index in range(len(self.weights)-1, -1, -1):
            weight_grad = outputs[index].T @ delta/batch_size
            bias_grad = delta.sum(axis=0, keepdims=True)/batch_size
            if index > 0:
                delta = (delta @ self.weights[index].T)
                delta = delta*self._derivative(outputs[index])
            self.weights[index] -= learning_rate*weight_grad
            self.biases[index] -= learning_rate*bias_grad

    def train(self, inputs, targets, learning_rate, epochs, batch_size):
        """обучает сеть на данных и записывает ошибки обучения
        параметры: input: данные, на которых обучается сеть
        targets: правильные ответы для этих даннх
        learning_rate: скорость изменения весов
        epochs: количество повторений обучения
        batch_size: количество примеров за один шаг обучения"""
        if learning_rate <= 0 or epochs < 1 or batch_size < 1:
            raise InvalidTrainingParamsError(
                "скорость, эпохи и размер батча должны быть больше 0")
        inputs = self._check_inputs(inputs)
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (len(inputs), self.layer_sizes[-1]):
            raise MismatchedDataError(
                "ответы не подходят: не совпадает число строк или выходов")
        generator = np.random.default_rng()
        report_step = max(1, epochs//10)
        for epoch in range(1, epochs+1):
            order = generator.permutation(len(inputs))
            for start in range(0, len(inputs), batch_size):
                batch = order[start:start+batch_size]
                outputs = self.forward(inputs[batch])
                self.backward(outputs, targets[batch], learning_rate)
            loss = float(np.mean((self.predict(inputs)-targets)**2))
            self.loss_history.append(loss)
            if epoch == 1 or epoch % report_step == 0:
                print(f"Эпоха {epoch}/{epochs}, ошибка: {loss:.6f}")

    def save(self, path):
        """сохраняет настройки сети и ее веса в файл
        параметры: path: путь, по которому нужно сохранить файд"""
        data = {"layer_sizes": self.layer_sizes,
                "activation": self.activation_name,
                "weights": [weight.tolist() for weight in self.weights],
                "biases": [bias.tolist() for bias in self.biases]}
        try:
            Path(path).write_text(json.dumps(data), encoding="utf-8")
        except OSError as error:
            raise DataFileError(
                f"не удалось сохранить файл: {error}") from error

    @classmethod
    def load(cls, path):
        """загружает сохраненную нейросеть из файла
        параметры: path: путь к файлу с настройками и весами сети
        возвращает: загруженную нейросеть"""
        try:
            text = Path(path).read_text(encoding="utf-8")
            data = json.loads(text)
            network = cls(data["layer_sizes"], data["activation"])
            network.weights = [np.array(item) for item in data["weights"]]
            network.biases = [np.array(item) for item in data["biases"]]
        except FileNotFoundError as error:
            raise DataFileError(f"файл не найден: {path}") from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DataFileError(
                f"файл поврежден или не подходит:{error}") from error
        return network
