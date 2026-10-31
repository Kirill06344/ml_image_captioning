"""Залить подготовленные COCO-файлы приватным датасетом и проверить приватность.

    .venv/bin/python tools/kaggle_upload.py --dir data/processed --slug coco2014-caption-5k --user ACCOUNT

Вложенные папки изображений передаются через --dir-mode zip. До загрузки проверить условия использования COCO.
Значения токенов не копируются в проект и не печатаются.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.kaggle_start import check_account, connect, check_remote_data
from src.data import Corpus
KAGGLE = str(ROOT / '.venv' / 'bin' / 'kaggle')


def run(*args: str, check: bool = True) -> str:
    # Глобальный KAGGLE_API_TOKEN молча увёл бы операцию на другой аккаунт.
    env = {k: v for k, v in os.environ.items() if k != 'KAGGLE_API_TOKEN'}
    result = subprocess.run([KAGGLE, *args], capture_output=True, text=True, env=env)
    output = (result.stdout or '') + (result.stderr or '')
    if check and result.returncode != 0:
        raise SystemExit(f'kaggle {" ".join(args)} упал:\n{output}')
    return output


def verify_private(dataset: str) -> None:
    """`datasets list --mine` индексируется с задержкой, поэтому источник правды — `datasets metadata`.

    Сразу после заливки metadata отвечает ошибкой: это «пока не готов», а не «не приватный». Поэтому повторяем.
    """
    for attempt in range(40):
        with tempfile.TemporaryDirectory() as tmp:
            result = run('datasets', 'metadata', '-p', tmp, dataset, check=False)
            path = Path(tmp) / 'dataset-metadata.json'
            if path.exists():
                info = json.loads(path.read_text())
                info = info.get('info', info)
                if info.get('isPrivate') is True:
                    print('приватность подтверждена сервером')
                    return
                raise SystemExit(f'!!! {dataset}: isPrivate={info.get("isPrivate")!r}, проверить на сайте и удалить')
        print(f'metadata ещё не отвечает ({attempt + 1}/40): {result.strip()[:120]}', flush=True)
        time.sleep(30)
    raise SystemExit(f'!!! не удалось проверить приватность {dataset}: проверить вручную')


def owned_dataset_exists(api, dataset):
    """Enumerate the authenticated owner's datasets; status is unsuitable for a new slug.

    Search indexes can lag. If an existing slug is absent from the list, the create endpoint
    still rejects it; this function never authorizes a version update.
    """
    for page in range(1, 1001):
        try:
            rows = api.dataset_list(mine=True, page=page)
        except Exception as exc:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            detail = f'HTTP {code}' if code is not None else type(exc).__name__
            raise RuntimeError(f'Не удалось получить список собственных датасетов ({detail}); загрузка не начата.') from exc
        if rows is None:
            raise RuntimeError('Список собственных датасетов не получен; загрузка не начата.')
        if any(getattr(row, 'ref', '').casefold() == dataset.casefold() for row in rows):
            return True
        if not rows:
            return False
    raise RuntimeError('Не завершено чтение списка датасетов; загрузка не начата.')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--message', default='', help='сообщение новой версии; с ним обновляется существующий датасет')
    args = parser.parse_args(argv)
    # Verify identity before creating metadata or performing any remote mutation.
    api = connect()
    check_account(api, args.user)
    corpus = Corpus(args.dir)
    dataset = f'{args.user}/{args.slug}'
    if not args.message and owned_dataset_exists(api, dataset) and check_remote_data(api, dataset, corpus):
        print('Приватный датасет уже загружен, manifest совпадает:', dataset)
        return
    (args.dir / 'dataset-metadata.json').write_text(json.dumps({
        'title': args.slug.replace('-', ' '), 'id': dataset, 'licenses': [{'name': 'other'}]}, indent=2))
    size = sum(f.stat().st_size for f in args.dir.rglob('*') if f.is_file()) / 2**20
    print(f'{dataset}: {size:.0f} МБ', flush=True)
    started = time.monotonic()
    # Без --public: датасет создаётся приватным. CLI отвечает кодом 0 и на ошибку создания, поэтому текст проверяется.
    output = (run('datasets', 'version', '-p', str(args.dir), '-m', args.message, '-q', '--keep-tabular', '--dir-mode', 'zip') if args.message
              else run('datasets', 'create', '-p', str(args.dir), '-q', '--keep-tabular', '--dir-mode', 'zip'))
    print(output)
    if 'error' in output.lower():
        raise SystemExit(f'{dataset}: заливка не удалась')
    print(f'заливка: {(time.monotonic() - started) / 60:.1f} мин')
    verify_private(dataset)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            print(str(exc), file=sys.stderr)
        else:
            print(f'Загрузка остановлена ({type(exc).__name__}). Проверьте сеть и вход в указанный аккаунт Kaggle.', file=sys.stderr)
        raise SystemExit(1)
