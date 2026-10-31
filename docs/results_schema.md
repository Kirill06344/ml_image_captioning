# Контракт оценки

Каждый прогон сохраняет predictions/<split>.jsonl: image_id, image (относительный путь), prediction,
references (список всех эталонов). Один прогноз на image_id; тесты на дубли и полноту обязательны.

scores_val.json и scores_test.json должны содержать:

```json
{
  "run": "slug-of-real-run",
  "role": "candidate",
  "split": "val",
  "n_images": 500,
  "dataset_fingerprint": "sha256-of-manifest",
  "image_ids_sha256": "sha256-of-sorted-scored-ids",
  "metric_units": {"CIDEr": "raw", "BLEU_4": "raw"},
  "scorer": {"implementation": "chosen-library", "version": "pinned-version", "tokenizer": "chosen-tokenizer"},
  "metrics": {"CIDEr": 0.0, "BLEU_4": 0.0}
}
```

Нули выше показывают только тип полей; это НЕ результаты и не файл для runs/.
role: candidate, reference или baseline. Имена метрик, шкалы и scorer должны совпадать у сравниваемых прогонов.
CIDEr нельзя смешивать между шкалами raw/×100. Подробности per-image для bootstrap хранить отдельно.
При выборе по val tools/collect_results.py записывает runs/selection.json. На test он принимает
только выбранную модель и baseline, уже присутствовавшие в зафиксированном сравнении val.
