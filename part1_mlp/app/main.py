"""консольное меню для работы с нейросетью"""
import numpy as np
from app.exceptions import (InvalidLayerSizeError,
                            InvalidTrainingParamsError,
                            NetworkNotReadyError,
                            NeuralNetworkError)

from app.managar import DatasetManager
from app.models import NeuralNetwork
from app.utils import (ask_float,
                       ask_int,
                       ask_layer_sizes,
                       ask_numbers,
                       clean_path,
                       load_rows)

MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 100
ACTIVATION_NAMES = {"1": "sigmoid", "2": "relu"}
DEFAULT_TEST_RATIO = 0.2
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 8
GRAPH_ROWS = 20
GRAPH_WIDTH = 40

MENU_TEXT = """__нейросеть__
               1. создать сеть
               2. загрузить данные из CSV
               3. обучить сеть
               4. сделать предсказание
               5. сохранить/загрузить веса
               6. показать график ошибки
               7. выйти"""


class ConsoleApp:
    """консольное меню для работы с нейросетью"""

    def __init__(self):
        """создает приложение без сети и данных"""
        self.network = None
        self.data = DatasetManager()

    def _require_network(self):
        """проверяет, создана ли сеть"""
        if self.network is None:
            raise NetworkNotReadyError("сначала создайте сеть.")

    def _require_split(self):
        """проверяет, загружены ли данные и разделены ли они"""
        if self.data.train_features is None:
            raise NetworkNotReadyError("сначала загрузите данные из CSV")

    def create_network(self):
        """создает сеть по размерам слоёв и активации"""
        print("нужно минимум 3 слоя: вход, скрытый, выход")
        print(f"размер каждого слоя: от {MIN_LAYER_SIZE} "
              f"до {MAX_LAYER_SIZE}.")
        print("первое число - число признаков в CSV, последнее - 1")
        sizes = ask_layer_sizes(
            "размеры слоев через запятую (например 4,4,1): ")
        if any(not MIN_LAYER_SIZE <= size <= MAX_LAYER_SIZE
               for size in sizes):
            raise InvalidLayerSizeError(
                f"размер слоя - от {MIN_LAYER_SIZE} до {MAX_LAYER_SIZE}")
        choice = input(
            "функция активации (1 - sigmoid, 2 - relu): ").strip()
        if choice not in ACTIVATION_NAMES:
            raise InvalidTrainingParamsError("нужно ввести 1 или 2")
        activation = ACTIVATION_NAMES[choice]
        self.network = NeuralNetwork(sizes, activation)
        print(f"сеть создана: слои {sizes}, активация {activation}")
        print("дальше: пункт 2 - загрузить данные, потом пункт 3 - обучить")

    def load_data(self):
        """загружает CSV, нормализует и делит на две группы"""
        path = clean_path(input("введите путь к CSV-файлу: "))
        self.data.load_csv(path)
        ratio = ask_float(
            f"Доля теста от 0.05 до 0.9 (Enter = {DEFAULT_TEST_RATIO}): ",
            DEFAULT_TEST_RATIO)
        self.data.split(ratio)
        print(
            f"данные загружены: признаков {self.data.features.shape[1]}, "
            f"обучение {len(self.data.train_features)} строк, "
            f"тест {len(self.data.test_features)} строк.")

    def train_network(self):
        """обучает сеть и показывает ошибку на тесте"""
        self._require_network()
        self._require_split()
        rate = ask_float(
            f"Скорость обучения от 0.01 до 1 "
            f"(Enter = {DEFAULT_LEARNING_RATE}): ",
            DEFAULT_LEARNING_RATE)
        epochs = ask_int(
            f"Число эпох от 10 до 5000 (Enter = {DEFAULT_EPOCHS}): ",
            DEFAULT_EPOCHS)
        batch_size = ask_int(
            f"Размер батча от 1 до 24 (Enter = {DEFAULT_BATCH_SIZE}): ",
            DEFAULT_BATCH_SIZE)
        self.network.loss_history = []
        self.network.train(
            self.data.train_features, self.data.train_targets,
            rate, epochs, batch_size)
        answers = self.network.predict(self.data.test_features)
        test_loss = float(np.mean((answers - self.data.test_targets) ** 2))
        correct = (answers > 0.5) == (self.data.test_targets > 0.5)
        print(
            f"обучение завершено, "
            f"ошибка на проверочных данных: {test_loss:.6f}, "
            f"верных ответов: {float(np.mean(correct)) * 100:.1f}%")

    def predict(self):
        """делает предсказания для данных введенных вручную или из файла"""
        self._require_network()
        if self.data.mean is None:
            raise NetworkNotReadyError(
                "загрузите данные в (пункт 2): ")
        mode = input("1 - ввести вручную, 2 - из файла: ").strip()
        if mode == "1":
            count = self.data.mean.shape[0]
            rows = [ask_numbers(f"введите {count} признаков через пробел: ")]
        elif mode == "2":
            rows = load_rows(clean_path(input("путь к файлу с примерами: ")))
        else:
            raise InvalidTrainingParamsError("нужно ввести 1 или 2.")
        normalized = self.data.transform(rows)
        answers = self.network.predict(normalized)
        for number, value in enumerate(answers[:, 0], start=1):
            print(f"пример {number}: ответ сети {value:.4f} "
                  f"(класс {int(value > 0.5)})")

    def save_or_load_weights(self):
        """сохраняет или загружает веса из файла"""
        choice = input("1 - сохранить веса, 2 - загрузить веса: ").strip()
        if choice not in ("1", "2"):
            raise InvalidTrainingParamsError("нужно ввести 1 или 2.")
        path = clean_path(input("имя файла (например weights.json): "))
        if choice == "1":
            self._require_network()
            self.network.save(path)
            print(f"веса сохранены в файл {path}.")
        else:
            self.network = NeuralNetwork.load(path)
            print(f"веса загружены, слои сети: {self.network.layer_sizes}.")

    def show_loss_graph(self):
        """рисует график ошибки"""
        self._require_network()
        history = self.network.loss_history
        if not history:
            raise NetworkNotReadyError(
                "история ошибок пуста, обучите сеть (пункт 3).")
        step = max(1, len(history) // GRAPH_ROWS)
        shown = history[::step]
        top = max(shown) or 1.0
        for number, loss in enumerate(shown):
            bar = "#" * int(GRAPH_WIDTH * loss / top)
            print(f"эпоха {number * step + 1:>5} | {bar} {loss:.5f}")

    def run(self):
        """запускает главный цикл меню"""
        actions = {
            "1": self.create_network,
            "2": self.load_data,
            "3": self.train_network,
            "4": self.predict,
            "5": self.save_or_load_weights,
            "6": self.show_loss_graph,
        }
        while True:
            print(MENU_TEXT)
            choice = input("выберите пункт: ").strip()
            if choice == "7":
                print("программа завершена")
                break
            action = actions.get(choice)
            if action is None:
                print("такого  пути нет, введите число от 1 до 7")
                continue
            try:
                action()
            except NeuralNetworkError as error:
                print(f"ошибка: {error}")


def main():
    """запускается программу"""
    ConsoleApp().run()


if __name__ == "__main__":
    main()
