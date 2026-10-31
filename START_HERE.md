# Проект завершён

Автор — Тесленко Анна Руслановна. Тема — Image captioning на MS-COCO 2014, подмножество 5k.
Соседний ml_text_summarization не менять. Использовать собственную .venv.

## Итог

- Подготовлено 5000 изображений: 4000/500/500, все подписи одного image_id в одном сплите.
- Обучен **один** рецепт: замороженный ResNet18 ImageNet + Transformer 2 слоя, seed=1.
- Полный запуск badzhik/caption-control завершён: 8 эпох, 5008 обновлений, лучший checkpoint — эпоха 5.
  Tesla T4, 486.1 с по таймеру тренера, peak allocated PyTorch 286.1 МиБ. Новое обучение не запускать.
- Val CIDEr: модель 0.6658, retrieval 0.4155, constant 0.0585.
- Test CIDEr: модель **0.6390**, retrieval **0.3846**, constant 0.0649.
  Прирост к retrieval — 66.2%, BLEU-4 модели — 0.1921, METEOR — 0.1972, ROUGE-L — 0.4550.
- Выбор зафиксирован по val в runs/selection.json до test. Test модели и бейзлайнов рассчитан локально.
- Проверены 3000 прогнозов (три метода × два сплита), все исходные эталоны и сопоставимость scorer.
- Полный pytest — 71 passed. METEOR стабилизирован для ru_RU.UTF-8 через локаль дочерней JVM en_US.
- Итоговые README, DOCX и PDF собраны. Подписи таблиц/рисунков снизу, оглавление обновлено.
  Рендер — LibreOffice 7.4 в Docker с Writer/Math/UNO и настоящими Times New Roman.

## Артефакты

- Отчёт: README.md; docs/report/Курсовая работа — Тесленко.docx и одноимённый PDF.
- Модель: runs/caption-control/best.pt, словарь и encoder/decoder находятся в checkpoint.
- Прогнозы: runs/{caption-control,caption-retrieval,caption-constant}/predictions/{val,test}.jsonl.
- Оценки: scores_val.json, scores_test.json и per_image_val/test.json в тех же каталогах.
- Таблицы, графики, bootstrap, галерея: docs/results/; вывод таблиц — tools/collect_results.py.
- Проверки: docs/validation/gpu_smoke.json, full_training.json и final_checks.json.

Генерация подписи к своей фотографии:

```bash
.venv/bin/python tools/predict.py --image photo.jpg --checkpoint runs/caption-control/best.pt
```

Группа 5140904/50101 и руководитель Селин Иван Андреевич указаны в project.yaml и на титуле. В основном тексте отчёта персональные сведения не дублируются.
Методические ограничения — один seed, один обучающий рецепт, собственный test 500, не Karpathy test 5k.
Сведения о предобучении и точные параметры — docs/experiment_protocol.md; команды — docs/runbook.md.

## Kaggle

Данные и код размещены приватно у badzhik. Токен — ~/.kaggle/access_token, в проект его не копировать.
Для uliabzh использовался ~/.kaggle/ul_access_token; ранний queued smoke не включён в результаты,
его отмена отдельно не подтверждена. Датасет кода uliabzh не менять при неизвестном статусе.
Сетевой доступ к Kaggle из терминала агента недоступен; пользователь скачал оба успешных прогона.
Новых GPU-запусков для завершённой работы не требуется. Все результаты сохранены локально.
