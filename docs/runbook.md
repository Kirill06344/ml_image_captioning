# Использование завершённого проекта

Обучение и окончательная оценка уже выполнены. Новый GPU-запуск для использования модели не требуется.
Автор и параметры отчёта — project.yaml; результаты — runs/*/scores_val/test.json.

## Подпись к своей фотографии

```bash
.venv/bin/python tools/predict.py --image photo.jpg --checkpoint runs/caption-control/best.pt
```

Checkpoint включает визуальный кодировщик, декодер, словарь и архитектуру. Загрузка дополнительных
ImageNet-весов при таком inference не требуется. CPU поддерживается; CUDA используется при наличии.

## Окружение

Из корня проекта, с собственной .venv:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/check_environment.py
.venv/bin/python -m pytest
```

Для CUDA нужны соответствующие официальные wheels PyTorch. На Kaggle использовались версии
образа: torch 2.11.0+cu128, torchvision 0.26.0+cu128. Локальная оценка test выполнена на CPU
с torch 2.6.0+cpu и torchvision 0.21.0+cpu. Все версии сохранены в run/evaluation_test.json.
Для метрик нужна Java. Локаль METEOR фиксируется внутри дочерней JVM, системную локаль менять не нужно.
Последний полный pytest: 71 passed. Не использовать .venv соседнего проекта.

## Данные и результаты

Официальный COCO уже подготовлен в data/processed. Разбиение и отбор — docs/experiment_protocol.md.
Для восстановления данных из источников:

```bash
.venv/bin/python tools/download_assets.py
```

Скрипт загружает аннотации и только выбранные 5000 изображений, затем ImageNet-веса.
Завершённые загрузки проверяются по SHA-256; незавершённый файл скачивается заново.
Готовая выборка не перезаписывается. Для уже распакованных исходников есть tools/prepare_data.py.

Основные артефакты:

- runs/caption-control/best.pt — checkpoint 5-й эпохи из 8 выполненных;
- runs/caption-control/history.json и run.json — реальные шаги, loss, время и память;
- runs/selection.json — выбор по val до test;
- predictions/val.jsonl и test.jsonl — по 500 прогнозов на метод;
- scores_val/test.json и per_image_val/test.json — оценки и индивидуальные метрики;
- docs/results/ — таблицы, графики, bootstrap и шесть примеров разных рангов CIDEr.

Повторная оценка выбранного checkpoint без обучения:

```bash
.venv/bin/python -m src.train --data data/processed --out runs/caption-control --run caption-control --model resnet18-transformer --eval-only --checkpoint runs/caption-control/best.pt --split test --selection runs/selection.json
```

Бейзлайны test также уже рассчитаны. Для воспроизведения:

```bash
.venv/bin/python tools/baselines.py --data data/processed --kind constant --run caption-constant --out runs/caption-constant --split test --selection runs/selection.json
.venv/bin/python tools/baselines.py --data data/processed --kind retrieval --run caption-retrieval --out runs/caption-retrieval --split test --selection runs/selection.json
```

Retrieval использует исходные замороженные ImageNet-признаки, первую подпись ближайшего train-соседа
и не требует обучения декодера. Все подписи val/test используются исключительно для оценки.

Сводные таблицы и галерея:

```bash
.venv/bin/python tools/collect_results.py --split val
.venv/bin/python tools/collect_results.py --split test
.venv/bin/python tools/build_examples.py --data data/processed --run runs/caption-control --split test
```

Выбор по val уже зафиксирован. Изменять его после просмотра test нельзя. Если val-оценки изменены,
сборщик остановится вместо молчаливого переопределения выбора.

## Kaggle: получение существующих артефактов

Владелец и исполнитель полного запуска — badzhik. Данные приватны:
badzhik/coco2014-caption-5k; код — badzhik/caption-code. Токен хранится вне проекта.

```bash
.venv/bin/python tools/kaggle_train.py --config configs/experiments/control.yaml --user badzhik --token-file ~/.kaggle/access_token --fetch-only
```

Это только скачивание. Для диагностики существующего запуска:

```bash
.venv/bin/python tools/kaggle_start.py --user badzhik --config configs/experiments/control.yaml --token-file ~/.kaggle/access_token --diagnose
```

Согласованный обучающий рецепт один. Архитектура ResNet18 + Transformer 2 слоя, batch=32,
seed=1, максимум 10 эпох и patience=3. Фактически пройдено 8 эпох, 5008 обновлений.
CLI --timeout 7200 ограничивал выполнение ядра, а timeout ожидания launcher сам по себе его не отменяет.
При неизвестном статусе повторный push запрещён; заранее сохраняется журнал попытки.
Режим --submit-only завершает локальную команду после подтверждённой отправки и не создаёт
повторный запуск при queued/running/complete. Завершение локального ожидания Ctrl+C не отменяет Kaggle.
Ранний smoke на uliabzh не участвовал в результатах; его код не обновлять при неизвестном статусе.

## Отчёт и рендер

Актуальные файлы — docs/report/Курсовая работа — Тесленко.docx и одноимённый PDF.
После изменения README или project.yaml экспорт и рендер нужно выполнить заново:

```bash
.venv/bin/python tools/export_docx.py
/usr/bin/python3 tools/render_docx.py --input 'docs/report/Курсовая работа — Тесленко.docx'
```

Второй команде нужны LibreOffice Writer/Math и python3-uno в системе. На этой машине рендер
выполнен в Docker, поскольку на хосте этих компонентов нет. Протестированный способ:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -v /usr/share/fonts/truetype/msttcorefonts:/usr/share/fonts/truetype/msttcorefonts:ro \
  -w /workspace lo-render sh -c \
  'apt-get update && apt-get install -y --no-install-recommends libreoffice-writer libreoffice-math python3-uno && fc-cache -f && python3 tools/render_docx.py --input "docs/report/Курсовая работа — Тесленко.docx" && chown -R 1000:1000 docs/report'
```

Образ lo-render сам по себе не включал Writer/Math/UNO: они установлены внутри временного контейнера.
Монтируются настоящие Times New Roman. Рендер обновляет оглавление и номера страниц, сохраняет DOCX/PDF.
После рендера проверить страницы визуально; текущая финальная версия содержит 20 страниц,
четыре таблицы и девять рисунков, включая все шесть фотографий приложения.
