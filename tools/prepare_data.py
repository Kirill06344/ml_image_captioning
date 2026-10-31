"""Собрать локальную 5k-выборку COCO captions, без скачиваний и сети."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_annotations(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    images = {int(row['id']): row for row in raw['images']}
    if len(images) != len(raw['images']):
        raise ValueError('Повторные image_id в annotations.images')
    references = {key: [] for key in images}
    for annotation in raw['annotations']:
        key = int(annotation['image_id'])
        if key not in images:
            raise ValueError(f'Подпись ссылается на отсутствующий image_id={key}')
        text = str(annotation['caption']).strip()
        if not text:
            raise ValueError(f'Пустая подпись image_id={key}')
        references[key].append(text)
    if any(not refs for refs in references.values()):
        raise ValueError('Изображение без эталонных подписей')
    return images, references


def stable_ids(keys, count):
    if count <= 0 or len(keys) < count:
        raise ValueError(f'Нужно {count} изображений, доступно {len(keys)}')
    return sorted(keys, key=lambda key: hashlib.sha256(str(key).encode()).hexdigest())[:count]


def prepare(train_annotations, val_annotations, images_root, out, train_n=4000, val_n=500, test_n=500):
    images_root, out = Path(images_root), Path(out)
    if any(count <= 0 for count in (train_n, val_n, test_n)):
        raise ValueError('Размер каждого сплита должен быть положительным')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out}: каталог не пуст; используйте новый путь, чтобы не менять готовую выборку')
    train_images, train_refs = load_annotations(train_annotations)
    held_images, held_refs = load_annotations(val_annotations)
    train_ids = stable_ids(train_images, train_n)
    held_ids = stable_ids(held_images, val_n + test_n)
    splits = {'train': train_ids, 'val': held_ids[:val_n], 'test': held_ids[val_n:]}
    if set(train_ids) & set(held_ids):
        raise ValueError('Пересечение train и val/test по image_id')
    planned = {}
    # Проверить все файлы до первой записи: частичный набор не выглядит готовым.
    for split, keys in splits.items():
        images, refs = (train_images, train_refs) if split == 'train' else (held_images, held_refs)
        source_split = 'train2014' if split == 'train' else 'val2014'
        planned[split] = []
        for key in keys:
            name = images[key]['file_name']
            if Path(name).name != name:
                raise ValueError(f'Некорректное имя файла: {name}')
            candidates = [images_root / source_split / name, images_root / name]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(f'{source_split}/{name}')
            relative = Path('images') / source_split / name
            planned[split].append((source, relative, {'image_id': key, 'image': relative.as_posix(),
                                                      'image_sha256': fingerprint(source),
                                                      'references': refs[key]}))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': 'MS-COCO 2014', 'selection': 'sha256(image_id), deterministic',
                'total_images': sum(map(len, splits.values())),
                'source_sha256': {'train_annotations': fingerprint(train_annotations),
                                  'val_annotations': fingerprint(val_annotations)}, 'splits': {}}
    for split, rows in planned.items():
        with (out / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
            for source, relative, row in rows:
                target = out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest['splits'][split] = {'n_images': len(rows), 'image_ids': splits[split],
                                     'rows_sha256': fingerprint(out / f'{split}.jsonl')}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-annotations', type=Path, required=True)
    parser.add_argument('--val-annotations', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/processed'))
    parser.add_argument('--train-size', type=int, default=4000)
    parser.add_argument('--val-size', type=int, default=500)
    parser.add_argument('--test-size', type=int, default=500)
    args = parser.parse_args()
    result = prepare(args.train_annotations, args.val_annotations, args.images, args.out,
                     args.train_size, args.val_size, args.test_size)
    print('Подготовлено изображений:', result['total_images'], 'manifest:', result['fingerprint'])


if __name__ == '__main__':
    main()
