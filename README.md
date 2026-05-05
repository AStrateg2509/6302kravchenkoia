# metetl

MET ETL — подготовка, скачивание+обработка и анализ коллекции Metropolitan Museum of Art.

Лабораторные работы №1–5 курса (ИКСИ-С, гр. 6302).

## Установка

С TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    metetl-kravchenko-6302
```

Из исходников (editable):

```bash
pip install -e .
```

Сборка артефактов:

```bash
pip install build
python -m build  # → dist/*.whl, dist/*.tar.gz
```

## Использование

```bash
metetl --help
metetl prepare --csv data/MetObjects.csv --output data/to_download.json --limit 100
metetl process --input data/to_download.json --output images --num 5 --impl numpy
metetl process --input data/to_download.json --output images --num 5 --streaming --impl c
metetl analyze --csv data/MetObjects.csv --output-dir data/plots
```

Параметры:

* `--impl {numpy,opencv,c}` — реализация свёртки в воркере;
  `c` использует собранное расширение `metetl.images.c_ext.image_ops`.
* `--streaming` — конвейер на async-генераторах (Lab 4 бонус).
* `-v` — DEBUG-вывод в консоль.
* `--log-file PATH` — переопределить путь к файлу лога (по умолчанию `logs/metetl.log`).

## Структура

```
src/metetl/
├── cli.py                 # консольный интерфейс
├── decorators.py          # @timeit
├── logging_config.py      # настройка логирования
├── analysis/
│   ├── data_to_download.py    # CSV → JSON
│   └── aggregations.py        # анализ + графики
└── images/
    ├── models.py              # Artwork, ColorArtwork, GrayscaleArtwork
    ├── processing.py          # ImageProcessor + async/streaming
    └── c_ext/                 # C-расширение image_ops
```

Логи: подробные в `logs/metetl.log` (DEBUG), краткие — в консоль (INFO).

## Тесты

```bash
python -m unittest discover -s tests -v
```
