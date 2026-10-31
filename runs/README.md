# Результаты

caption-control — единственный полный обучающий прогон, checkpoint 5-й эпохи из 8 выполненных.
caption-constant и caption-retrieval — бейзлайны, без обучения captioning-декодера.
selection.json фиксирует выбор по val до test. В каждом каталоге результатов находятся
scores_val/test.json, per_image_val/test.json и predictions/val/test.jsonl.

caption-smoke-main — успешная проверка на GPU. local-coco-smoke и local-synthetic-smoke —
локальные технические проверки; их smoke_scores.json не включаются в сравнительные таблицы.
Ранний caption-smoke на uliabzh не использовался для выбора модели и итоговых метрик.

Подробности — ../docs/experiment_protocol.md; сводные данные — ../docs/results/.
Данные и веса исключены из git; новый remote и push не выполнялись.
