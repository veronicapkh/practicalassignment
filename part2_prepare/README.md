# xасть 2. подготовка данных

утилита с командами `prepare` (собирает картинки в массив `.npy`) и `doctor` (проверяет окружение).

## Запуск

из корня проекта:

```
cd part2_prepare
python -m app.cli doctor
python -m app.cli prepare
python -m app.cli prepare --ext jpg --size 16 --output ..\data\output\small.npy
```

изображения лежат в `data/input/images/`.
