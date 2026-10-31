"""Build a gallery from actual scored predictions: low, median and high CIDEr."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Corpus
from src.evaluation import validate_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--out', type=Path, default=Path('docs/results'))
    args = p.parse_args()
    corpus = Corpus(args.data)
    scores = json.loads((args.run / f'scores_{args.split}.json').read_text())
    if scores['dataset_fingerprint'] != corpus.manifest['fingerprint']:
        raise ValueError('Другой manifest')
    records = [json.loads(s) for s in (args.run / 'predictions' / f'{args.split}.jsonl').read_text().splitlines()]
    source_rows = corpus.rows(args.split)
    source_by_id = {row['image_id']: row for row in source_rows}
    records = validate_predictions(source_rows, records)
    individual = json.loads((args.run / f'per_image_{args.split}.json').read_text())
    ranked = sorted(records, key=lambda r: (individual[str(r['image_id'])]['CIDEr'], r['image_id']))
    indices = sorted({0, 1, (len(ranked) - 1) // 2, len(ranked) // 2, len(ranked) - 2, len(ranked) - 1})
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ['# Примеры описаний', '',
             'Приведены два примера с наименьшим CIDEr, два из середины списка и два с наибольшей оценкой. При одинаковых значениях порядок определяется image_id.', '']
    for i in indices:
        row = ranked[i]
        name = f'example-{args.split}-{row["image_id"]}.jpg'
        corpus.image(source_by_id[row['image_id']]).save(args.out / name)
        # README and exporter resolve image paths relative to the project root.
        relative = (args.out / name).resolve().relative_to(Path.cwd()).as_posix()
        lines += [f'## Изображение {row["image_id"]}', '', f'![Изображение {row["image_id"]}]({relative})', '',
                  '**Предсказание:** ' + row['prediction'], '', '**Эталонные подписи:**', '',
                  *['- ' + text for text in row['references']], '']
    (args.out / 'examples.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
