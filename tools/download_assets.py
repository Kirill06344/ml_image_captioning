"""Download the selected official COCO images and ImageNet weights, then prepare the 5k manifest.

No Kaggle calls. Interrupted downloads can be retried with the same command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import Corpus, sha256, write_json
from tools.prepare_data import load_annotations, prepare, stable_ids

# HTTPS endpoint of the public S3 bucket used by images.cocodataset.org.
COCO = 'https://s3.amazonaws.com/images.cocodataset.org'
ANNOTATIONS = COCO + '/annotations/annotations_trainval2014.zip'
WEIGHTS = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


def image_valid(path):
    from PIL import Image
    with Image.open(path) as image:
        image.verify()


def download(url, path, *, timeout=60, expected_prefix=None, validator=None):
    """Atomic transfer; reuse only files verified by checksum and content checks."""
    path = Path(path)
    record = path.with_name(path.name + '.download.json')
    if path.is_file():
        digest = sha256(path)
        if expected_prefix and not digest.startswith(expected_prefix):
            raise ValueError(f'{path}: хэш не совпадает с официальным хэшем весов')
        if record.is_file():
            previous = json.loads(record.read_text())
            if previous['url'] != url or previous['sha256'] != digest:
                raise ValueError(f'{path}: существующий файл отличается от записи загрузки')
        elif not expected_prefix:
            raise ValueError(f'{path}: существующий файл без записи происхождения; используйте новый raw-каталог')
        if validator:
            validator(path)
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    last_error = None
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            request = urllib.request.Request(url, headers={'User-Agent': 'COCO-captioning-coursework/1.0'})
            with urllib.request.urlopen(request, timeout=timeout) as response, partial.open('wb') as stream:
                expected_size = response.headers.get('Content-Length')
                written = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    written += len(block)
                    digest.update(block)
                if expected_size is not None and written != int(expected_size):
                    raise ValueError('Не завершена загрузка: неверный Content-Length')
            value = digest.hexdigest()
            if expected_prefix and not value.startswith(expected_prefix):
                raise ValueError('Хэш загруженных весов не совпадает с официальным')
            if validator:
                validator(partial)
            # Persist provenance first; an interrupted rename is safe to retry.
            write_json(record, {'url': url, 'sha256': value, 'bytes': written})
            os.replace(partial, path)
            return value
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'Не удалось загрузить {url}: {last_error}') from last_error


def captions_valid(path):
    with zipfile.ZipFile(path) as archive:
        for split in ('train', 'val'):
            name = f'annotations/captions_{split}2014.json'
            # Reading validates ZIP CRC; parsing validates the expected annotation structure.
            raw = json.loads(archive.read(name))
            if not raw.get('images') or not raw.get('annotations'):
                raise ValueError(f'{name}: пустые аннотации')


def selected_images(train_annotations, val_annotations, train_n, val_n, test_n):
    train, _ = load_annotations(train_annotations)
    held, _ = load_annotations(val_annotations)
    train_ids, held_ids = stable_ids(train, train_n), stable_ids(held, val_n + test_n)
    if set(train_ids) & set(held_ids):
        raise ValueError('Пересечение train и val/test')
    selected = []
    for split, images, ids in [('train2014', train, train_ids), ('val2014', held, held_ids)]:
        for key in ids:
            name = images[key]['file_name']
            if Path(name).name != name:
                raise ValueError('Небезопасное имя файла COCO')
            selected.append((split, name))
    return selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT / 'data/raw')
    p.add_argument('--out', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--weights-dir', type=Path, default=ROOT / '.cache/torch/hub/checkpoints')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--plan', action='store_true', help='print sources and counts without any network or writes')
    args = p.parse_args(argv)
    if args.workers < 1 or args.timeout < 1:
        p.error('workers и timeout должны быть положительными')
    if args.plan:
        print('COCO: 4000 train2014 + 1000 val2014 → 4000/500/500; все подписи вместе.')
        print('Аннотации:', ANNOTATIONS)
        print('Фотографии: только 5000 выбранных файлов, не полные архивы train/val.')
        print('ImageNet ResNet18:', WEIGHTS)
        print('Результат:', args.out)
        print('Загрузок, обучения и обращений в Kaggle нет.')
        return
    if args.out.exists():
        if not (args.out / 'manifest.json').is_file():
            raise ValueError(f'{args.out}: нет готового manifest; используйте новый --out')
        corpus = Corpus(args.out)
        for split, n in [('train', 4000), ('val', 500), ('test', 500)]:
            rows = corpus.rows(split)
            if len(rows) != n:
                raise ValueError('Существующая выборка отличается от 4000/500/500')
            # Verify bytes via manifest without inspecting held captions for modeling.
            for row in rows:
                if sha256(corpus.root / row['image']) != row['image_sha256']:
                    raise ValueError(f'Изменилось изображение {row["image_id"]}')
        print('Готовая выборка проверена:', args.out, flush=True)
    else:
        archive_path = args.raw / 'annotations_trainval2014.zip'
        print('Загрузка официальных аннотаций (~241 МБ); затем только выбранных изображений.', flush=True)
        archive_digest = download(ANNOTATIONS, archive_path, timeout=args.timeout, validator=captions_valid)
        annotations = args.raw / 'annotations'
        annotations.mkdir(parents=True, exist_ok=True)
        paths = []
        with zipfile.ZipFile(archive_path) as archive:
            for split in ('train', 'val'):
                name = f'captions_{split}2014.json'
                target = annotations / name
                partial = target.with_name(name + '.part')
                with archive.open('annotations/' + name) as source, partial.open('wb') as stream:
                    shutil.copyfileobj(source, stream)
                os.replace(partial, target)
                paths.append(target)
        planned = selected_images(*paths, 4000, 500, 500)
        print('Фотографий к проверке/загрузке:', len(planned), flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download, f'{COCO}/{split}/{name}', args.raw / split / name,
                                       timeout=args.timeout, validator=image_valid) for split, name in planned]
            try:
                for index, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if index % 100 == 0 or index == len(planned):
                        print(f'Изображения: {index}/{len(planned)}', flush=True)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.prepare-coco-', dir=args.out.parent) as staging:
            prepared = Path(staging) / 'processed'
            manifest = prepare(*paths, args.raw, prepared)
            # Full source/transfer provenance stays beside raw data; manifest hashes the annotations/images.
            write_json(args.raw / 'sources.json', {'annotations_url': ANNOTATIONS, 'annotations_sha256': archive_digest,
                                                  'image_base_url': COCO, 'images': len(planned),
                                                  'manifest_fingerprint': manifest['fingerprint']})
            os.replace(prepared, args.out)
        print('Подготовлена выборка:', args.out, flush=True)
    target = args.weights_dir / 'resnet18-f37072fd.pth'
    print('Загрузка/проверка весов ResNet18 ImageNet-1K V1.', flush=True)
    download(WEIGHTS, target, timeout=args.timeout, expected_prefix='f37072fd')
    print('Готово. COCO 4000/500/500 и веса получены. Обучение не запускалось.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SystemExit(str(exc))
