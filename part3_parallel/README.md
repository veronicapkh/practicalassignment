# часть 3. потоки и процессы

потоки читают изображения, процессы делают из них изменённые копии. программа сравнивает время с обычной версией и записывает результат в `data/output/performance_report.json`.

## Запуск

из корня проекта:

```
cd part3_parallel
python -m app.main --path ..\data\input\images --copies 60
```
