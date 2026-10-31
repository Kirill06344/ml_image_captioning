"""Собрать настоящие оценки captioning в Markdown/CSV/PNG; метрики не вычисляет."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_scores(path, split):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if report['split'] != split or not isinstance(report['n_images'], int) or report['n_images'] <= 0:
        raise ValueError(f'{path}: неправильный split или объём выборки')
    if report.get('role', 'candidate') not in ('candidate', 'reference', 'baseline'):
        raise ValueError(f'{path}: неизвестная роль')
    for field in ['run', 'dataset_fingerprint', 'image_ids_sha256', 'scorer', 'metric_units']:
        if not report.get(field):
            raise ValueError(f'{path}: отсутствует {field}')
    if not report['metrics'] or set(report['metrics']) != set(report['metric_units']):
        raise ValueError(f'{path}: шкалы метрик не заданы полностью')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in report['metrics'].values()):
        raise ValueError(f'{path}: некорректные метрики')
    return report


def check_comparable(reports):
    for field in ['n_images', 'dataset_fingerprint', 'image_ids_sha256', 'scorer', 'metric_units']:
        values = [json.dumps(report[field], sort_keys=True) for report in reports]
        if len(set(values)) != 1:
            raise ValueError(f'Нельзя сравнивать: различается {field}')


def collect(root, split, primary, output):
    root, output = Path(root), Path(output)
    paths = sorted(root.glob(f'*/scores_{split}.json'))
    if not paths:
        raise ValueError('Нет реальных scores-файлов. Сначала реализовать scorer и оценить прогнозы.')
    reports = [load_scores(path, split) for path in paths]
    if len({r['run'] for r in reports}) != len(reports):
        raise ValueError('Повторные имена прогонов')
    check_comparable(reports)
    if primary not in reports[0]['metrics']:
        raise ValueError(f'Отсутствует главная метрика {primary}')
    selection_path = root / 'selection.json'
    if split == 'val':
        candidates = [r for r in reports if r.get('role', 'candidate') == 'candidate']
        if not candidates:
            raise ValueError('Нет моделей для выбора по val')
        winner = max(candidates, key=lambda r: r['metrics'][primary])
        selection = {'split': 'val', 'primary_metric': primary, 'final': winner['run'],
                     'roles': {r['run']: r.get('role', 'candidate') for r in reports},
                     'metric_units': winner['metric_units'], 'scorer': winner['scorer'],
                     'dataset_fingerprint': winner['dataset_fingerprint'],
                     'val_scores_sha256': {r['run']: hashlib.sha256(p.read_bytes()).hexdigest() for p, r in zip(paths, reports)}}
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError('Выбор по val уже зафиксирован и отличается; оформите новый эксперимент отдельно')
        selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        allowed = {selection['final']} | {r for r, role in selection['roles'].items() if role == 'baseline'}
        if not {r['run'] for r in reports} <= allowed:
            raise ValueError('Test разрешён только выбранной модели и зафиксированным бейзлайнам')
        for report in reports:
            for field in ['scorer', 'metric_units', 'dataset_fingerprint']:
                if report[field] != selection[field]:
                    raise ValueError(f'Test отличается от val: {field}')
    output.mkdir(parents=True, exist_ok=True)
    metrics = list(reports[0]['metrics'])
    headers = ['Прогон', 'Изображений', *metrics]
    rows = [[r['run'], str(r['n_images']), *[f'{r["metrics"][m]:.4f}' for m in metrics]] for r in reports]
    table = '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
                       *['| ' + ' | '.join(row) + ' |' for row in rows]]) + '\n'
    (output / f'{split}.md').write_text(table, encoding='utf-8')
    with (output / f'{split}.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'n_images', *metrics])
        for report in reports:
            writer.writerow([report['run'], report['n_images'], *[report['metrics'][m] for m in metrics]])
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (output / f'provenance_{split}.json').write_text(json.dumps({'selection': selection, 'inputs_sha256': sources},
                                                              ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return reports, selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    args = p.parse_args()
    project = yaml.safe_load((ROOT / 'project.yaml').read_text(encoding='utf-8'))
    output = ROOT / 'docs/results'
    reports, selection = collect(ROOT / 'runs', args.split, project['primary_metric'], output)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/captioning-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metric = project['primary_metric']
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    labels = {'caption-constant': 'Постоянная подпись',
              'caption-control': 'ResNet-18 + Transformer',
              'caption-retrieval': 'Ближайшее изображение'}
    ax.barh([labels.get(r['run'], r['run']) for r in reports], [r['metrics'][metric] for r in reports])
    ax.set(xlabel=metric, title='Проверочная выборка' if args.split == 'val' else 'Тестовая выборка')
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.title.set_size(14)
    fig.tight_layout()
    fig.savefig(output / f'{args.split}_{metric}.png', dpi=180)
    plt.close(fig)
    print('Готово:', output, 'выбор по val:', selection['final'])


if __name__ == '__main__':
    main()
