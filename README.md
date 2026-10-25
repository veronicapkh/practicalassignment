# (practicalassignment)

практическое задание «Основы нейронных сетей»: три программы на Python.

- `part1_mlp` - нейросеть с консольным меню.
- `part2_prepare` - утилита подготовки данных (команды `prepare` и `doctor`).
- `part3_parallel` - потоки и процессы для аугментации картинок.
- `data/input` - тестовые данные (CSV и картинки).

## Установка

нужен Python 3.12 или новее. установлена версия Python 3.15.

```
pip install -r requirements.txt
```

настройки можно менять через файл `.env` (образец - `.env.example`).

## Проверка стиля

```
python -m flake8 part1_mlp part2_prepare part3_parallel
```
