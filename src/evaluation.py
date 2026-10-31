"""COCO-compatible scoring with all references and strict coverage checks."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from .data import write_json

SCORER = {'implementation': 'pycocoevalcap', 'version': '1.2', 'meteor_java_locale': 'en_US',
          'tokenizer': 'Stanford PTBTokenizer 3.4.1, lowercase, COCO punctuation filtering'}


def validate_predictions(rows, predictions):
    expected = {r['image_id']: r for r in rows}
    found = {r['image_id']: r for r in predictions}
    if len(expected) != len(rows) or len(found) != len(predictions) or found.keys() != expected.keys():
        raise ValueError('Прогнозы: дубли или неполное покрытие image_id')
    ordered = []
    for row in rows:
        pred = found[row['image_id']]
        if pred['references'] != row['references'] or pred['image'] != row['image']:
            raise ValueError('Прогнозы: эталоны или путь не совпадают с manifest')
        if not isinstance(pred['prediction'], str):
            raise ValueError('Прогноз должен быть строкой')
        ordered.append(pred)
    return ordered


def coco_metrics(predictions, cider_only=False):
    if version('pycocoevalcap') != '1.2':
        raise ValueError('Нужен pycocoevalcap==1.2')
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    ids = [r['image_id'] for r in predictions]
    if len(ids) < 2:
        raise ValueError('CIDEr с corpus IDF требует хотя бы два изображения')
    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize({r['image_id']: [{'caption': t} for t in r['references']] for r in predictions})
    hyps = tokenizer.tokenize({r['image_id']: [{'caption': r['prediction']}] for r in predictions})
    scorers = [(Cider(), ['CIDEr'])]
    if not cider_only:
        from pycocoevalcap.bleu.bleu import Bleu
        from .meteor import CocoMeteor
        from pycocoevalcap.rouge.rouge import Rouge
        scorers += [(Bleu(4), ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']),
                    (CocoMeteor(), ['METEOR']), (Rouge(), ['ROUGE_L'])]
    metrics, individual = {}, {key: {} for key in ids}
    try:
        for scorer, names in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if len(names) == 1:
                score, scores = [score], [scores]
            for name, value, values in zip(names, score, scores):
                metrics[name] = float(value)
                if not math.isfinite(metrics[name]):
                    raise ValueError(f'Нечисловая метрика {name}')
                for key, val in zip(ids, values):
                    individual[key][name] = float(val)
    finally:
        for scorer, _ in scorers:
            if hasattr(scorer, 'close'):
                scorer.close()
    return metrics, individual


def require_test_selection(selection_path, run, role, fingerprint):
    if selection_path is None:
        raise ValueError('Test требует --selection, зафиксированный по val')
    selection = json.loads(Path(selection_path).read_text())
    allowed = run == selection['final'] and role == 'candidate'
    allowed |= selection['roles'].get(run) == role == 'baseline'
    if selection.get('split') != 'val' or selection['dataset_fingerprint'] != fingerprint or not allowed:
        raise ValueError('Test разрешён только выбранной модели и val-бейзлайнам')
    return selection


def score_predictions(corpus, split, path, out, run, role='candidate', selection=None):
    if split == 'test':
        selected = require_test_selection(selection, run, role, corpus.manifest['fingerprint'])
        if selected['scorer'] != SCORER:
            raise ValueError('Test scorer отличается от val')
    rows = corpus.rows(split)
    predictions = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    predictions = validate_predictions(rows, predictions)
    metrics, individual = coco_metrics(predictions)
    report = {'run': run, 'role': role, 'split': split, 'n_images': len(rows),
              'dataset_fingerprint': corpus.manifest['fingerprint'],
              'image_ids_sha256': hashlib.sha256(json.dumps(sorted(r['image_id'] for r in rows)).encode()).hexdigest(),
              'metric_units': {m: 'raw' for m in metrics}, 'scorer': SCORER, 'metrics': metrics}
    write_json(Path(out) / f'scores_{split}.json', report)
    write_json(Path(out) / f'per_image_{split}.json', individual)
    return report
