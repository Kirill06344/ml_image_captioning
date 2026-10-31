"""Запуск обучения на Kaggle: залить код, собрать ядро, дождаться, забрать результат.

Ядро типа `script` — один файл, а обучение — пакет `src/`. Поэтому пакет едет приватным датасетом, а ядро становится
стабом, который кладёт его на `sys.path` и зовёт `main([...])`. Чтобы не обучиться молча по старому коду, хэш пакета
передаётся ядру ключом `--expect-code-hash`, и трейнер падает при расхождении.

    .venv/bin/python tools/kaggle_train.py --config configs/experiments/smoke.yaml --dry-run
    .venv/bin/python tools/kaggle_train.py --config configs/experiments/smoke.yaml
    .venv/bin/python tools/kaggle_train.py --config configs/experiments/smoke.yaml --fetch-only
    .venv/bin/python tools/kaggle_train.py --config configs/experiments/smoke.yaml --attach

Ядро может идти на аккаунте участника команды, со своей квотой: `--runner-user` и `--runner-token` (файл его токена).
Код и данные остаются у `--user`, исполнителю они расшарены как Viewer. Ловушки — docs/pitfalls.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KAGGLE = str(ROOT / '.venv' / 'bin' / 'kaggle')
PACKAGE = ROOT / 'src'
STAGING = ROOT / 'runs' / '.kaggle'
AUTH_TOKEN_FILE = None

from src.codehash import package_files, package_hash  # noqa: E402
from src.config import load_config, train_argv  # noqa: E402


def run(*args: str, check: bool = True, token: str | None = None) -> str:
    """Без `token` — аккаунт по умолчанию; `token` — путь к файлу токена другого аккаунта.

    Глобальный KAGGLE_API_TOKEN молча увёл бы операции владельца на другой аккаунт, поэтому из окружения он убирается.
    """
    env = {k: v for k, v in os.environ.items() if k != 'KAGGLE_API_TOKEN'}
    selected_token_file = token or AUTH_TOKEN_FILE
    if selected_token_file:
        env['KAGGLE_API_TOKEN'] = selected_token_file
    result = subprocess.run([KAGGLE, *args], capture_output=True, text=True, env=env)
    output = (result.stdout or '') + (result.stderr or '')
    if check and result.returncode != 0:
        raise SystemExit(f'kaggle {" ".join(args)} упал:\n{output}')
    return output


def worker_status(output: str) -> str | None:
    """Статус ядра из ответа `kernels status`. None — статуса в ответе нет: сетевой сбой — это не падение ядра."""
    match = re.search(r'\bhas status [\"\'](?:KernelWorkerStatus\.)?([A-Za-z_]+)[\"\']', output)
    return match.group(1).lower() if match else None


def finished(status: str | None) -> bool:
    return status in ('complete', 'error') or (status or '').startswith('cancel')


def waiting_kernels(digest: str) -> list[str]:
    """Свои ядра в очереди, ждущие другой версии кода.

    Ядро читает датасет кода при старте, а не при отправке: новая версия, залитая поверх ожидающего ядра, уронила бы
    его проверку хэша.
    """
    waiting = []
    active = STAGING / 'active'
    for entry in sorted(active.glob('*')) if active.exists() else []:
        kernel, expects, *owner = entry.read_text().split()
        status = worker_status(run('kernels', 'status', kernel, check=False, token=owner[0] if owner else None))
        if finished(status):
            entry.unlink()
        elif status in ('queued', None) and expects != digest:
            waiting.append(kernel)
    return waiting


def upload_code(user: str, slug: str, runner_token: str | None = None) -> str:
    """Залить `src/` приватным датасетом одним zip: иначе `datasets create` пропускает подкаталоги."""
    digest = package_hash(PACKAGE)
    marker = STAGING / f'.{user}-{slug}.hash'
    dataset = f'{user}/{slug}'
    if marker.exists() and marker.read_text() == digest:
        print(f'код не менялся: {dataset}, хэш {digest}')
        return digest
    waiting = waiting_kernels(digest)
    if waiting:
        raise SystemExit(f'код изменился ({digest}), а в очереди ядра {waiting}: дождаться их старта')
    staging = STAGING / slug
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(staging / 'src.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in package_files(PACKAGE):
            archive.write(path, Path('src') / path.relative_to(PACKAGE))
    marker_name = f'code-{digest}.txt'
    (staging / marker_name).write_text(digest)
    (staging / 'dataset-metadata.json').write_text(json.dumps({
        'title': slug.replace('-', ' '), 'id': dataset, 'licenses': [{'name': 'other'}]}, indent=2))
    exists = dataset in run('datasets', 'list', '--mine', '-s', slug, check=False)
    # Без --public: датасет создаётся приватным. CLI отвечает кодом 0 и на ошибку создания.
    output = (run('datasets', 'version', '-p', str(staging), '-m', digest, '-q') if exists
              else run('datasets', 'create', '-p', str(staging), '-q'))
    print(output)
    if 'error' in output.lower():
        raise SystemExit(f'{dataset}: заливка кода не удалась')
    # Ядро, отправленное во время обработки версии, получило бы прежний код и упало бы на проверке хэша. `datasets status`
    # отвечает `ready` и про прошлую версию, поэтому признак готовности — файл с хэшем в имени в списке файлов.
    print('жду обработку версии кода', flush=True)
    for _ in range(40):
        time.sleep(15)
        if marker_name in run('datasets', 'files', dataset, check=False, token=runner_token):
            break
    else:
        raise SystemExit(f'{dataset}: версия не обработана за 10 мин')
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest)
    print(f'код залит: {dataset}, хэш {digest}')
    return digest


def build_kernel(folder: Path, runner: str, owner: str, slug: str, data_slug: str, code_slug: str, argv: list[str],
                 internet: bool, data_owner: str | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'run_training.py').write_text(
        '# === сгенерировано tools/kaggle_train.py ===\n'
        'import sys, zipfile, subprocess\n'
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'pycocoevalcap==1.2'], check=True)\n"
        'from pathlib import Path\n'
        '# Kaggle мог распаковать архив при монтировании, а мог оставить файлом: ищем дерево, потом zip.\n'
        "found = sorted(Path('/kaggle/input').rglob('src/train.py'))\n"
        'if found:\n'
        '    root = found[0].parents[1]\n'
        'else:\n'
        "    archives = sorted(Path('/kaggle/input').rglob('src.zip'))\n"
        '    if not archives:\n'
        "        raise SystemExit('пакет src не найден под /kaggle/input')\n"
        "    root = Path('/kaggle/working/code')\n"
        '    zipfile.ZipFile(archives[0]).extractall(root)\n'
        'sys.path.insert(0, str(root))\n'
        "print(f'пакет: {root}', flush=True)\n"
        'from src.train import main\n'
        '# Защита обязательна: процессы на GPU запускаются через spawn и импортируют этот файл заново.\n'
        "if __name__ == '__main__':\n"
        f'    main({argv!r})\n')
    (folder / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'{runner}/{slug}', 'title': slug.replace('-', ' '), 'code_file': 'run_training.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': internet, 'dataset_sources': [f'{data_owner or owner}/{data_slug}', f'{owner}/{code_slug}'],
        'competition_sources': [], 'kernel_sources': []}, indent=2))


def wait(kernel: str, poll: int, timeout_minutes: int, token: str | None = None) -> str:
    """Ждать конца ядра. Срок — по monotonic: эти часы стоят, пока ноутбук спит, и пробуждение не выглядит таймаутом."""
    deadline = time.monotonic() + timeout_minutes * 60
    last = ''
    while time.monotonic() < deadline:
        output = run('kernels', 'status', kernel, check=False, token=token).strip()
        status = worker_status(output)
        if status and status != last:
            print(f'[{time.strftime("%H:%M:%S")}] {status}', flush=True)
            last = status
        if finished(status):
            return status
        time.sleep(poll)
    raise SystemExit(f'ядро не завершилось за {timeout_minutes} мин: {last}. Не перезапускать: забрать --fetch-only')


def fetch(kernel: str, out: Path, token: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(run('kernels', 'output', kernel, '-p', str(out), '--force', token=token))
    for path in sorted(out.iterdir()):
        print(f'  {path.name}')


def push_and_track(kernel, digest, folder, out, timeout_minutes, token=None):
    """Keep a durable attempt record before push; confirm the server acknowledgement afterwards."""
    active = STAGING / 'active'
    active.mkdir(parents=True, exist_ok=True)
    (active / out.name).write_text(f'{kernel} {digest}' + (f' {token}' if token else ''))
    receipt = {'kernel': kernel, 'code_hash': digest, 'state': 'push_attempted'}
    (out / 'submission.json').write_text(json.dumps(receipt))
    output = run('kernels', 'push', '-p', str(folder), '--timeout', str(timeout_minutes * 60), token=token)
    print(output, flush=True)
    if not re.search(r'Kernel version(?: \d+)? successfully pushed', output, re.IGNORECASE):
        raise SystemExit('Kaggle не подтвердил отправку. Запись попытки сохранена; сначала выяснить статус, повторно не отправлять.')
    receipt.update({'state': 'submitted', 'submitted_at_utc': datetime.now(timezone.utc).isoformat()})
    (out / 'submission.json').write_text(json.dumps(receipt))
    return active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--user', default=None, help='владелец кода и данных')
    parser.add_argument('--token-file', type=Path, help='файл access token владельца кода (--user)')
    parser.add_argument('--data-owner', default=None, help='владелец общего датасета COCO; по умолчанию --user')
    parser.add_argument('--runner-user', default=None, help='аккаунт, на котором идёт ядро; по умолчанию --user')
    parser.add_argument('--runner-token', default=None, help='файл токена аккаунта-исполнителя')
    parser.add_argument('--code-slug', default='caption-code')
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='показать ключи ядра и выйти, ничего не заливая')
    parser.add_argument('--submit-only', action='store_true', help='отправить и выйти после подтверждения Kaggle, без ожидания и скачивания')
    parser.add_argument('--fetch-only', action='store_true', help='только забрать результат')
    parser.add_argument('--attach', action='store_true', help='дождаться уже отправленного ядра и забрать результат')
    parser.add_argument('--poll', type=int, default=60)
    args = parser.parse_args()
    if args.submit_only and (args.fetch_only or args.attach):
        parser.error('--submit-only нельзя сочетать с --fetch-only или --attach')
    global AUTH_TOKEN_FILE
    AUTH_TOKEN_FILE = None
    if args.token_file:
        path = args.token_file.expanduser().resolve()
        if not path.is_file():
            parser.error('--token-file: файл не найден')
        AUTH_TOKEN_FILE = str(path)

    if args.user is None:
        if args.dry_run:
            args.user = 'YOUR_KAGGLE_USER'
        else:
            parser.error('укажите --user: аккаунт не переносится из другого проекта')

    from src.train import build_parser, IMPLEMENTED

    config = load_config(args.config)
    slug = config['slug']
    kaggle = config['kaggle']
    if bool(args.runner_user) != bool(args.runner_token):
        raise SystemExit('--runner-user и --runner-token задаются только вместе')
    runner = args.runner_user or args.user
    token = str(Path(args.runner_token).expanduser()) if args.runner_token else AUTH_TOKEN_FILE
    kernel = f'{runner}/{slug}'
    out = ROOT / 'runs' / slug
    data_slug = kaggle['data_slug']
    timeout = int(kaggle.get('timeout_minutes', 780))
    if args.fetch_only:
        fetch(kernel, out, token)
        return
    if args.attach:
        status = wait(kernel, args.poll, timeout, token)
        (STAGING / 'active' / slug).unlink(missing_ok=True)
        fetch(kernel, out, token)
        if status != 'complete':
            raise SystemExit(f'ядро завершилось со статусом {status}: лог в {out}')
        return

    # Точный путь монтирования Kaggle менял: данные находит find_data по манифесту под /kaggle/input.
    argv = ['--data', '/kaggle/input', '--out', '/kaggle/working', '--run', slug, '--require-gpu',
            *train_argv(build_parser(), config['train'])]
    if args.dry_run:
        print(f'ядро {kernel}\nданные {data_slug}\nхэш кода {package_hash(PACKAGE)}\nлимит выполнения ядра {timeout * 60} с\nключи {" ".join(argv)}')
        return
    if not IMPLEMENTED:
        raise SystemExit('Тренер captioning ещё не реализован. Доступны --dry-run и получение существующих результатов; ничего не отправлено.')
    if not args.skip_tests:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-x', '-q', str(ROOT / 'tests')], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f'тесты упали, ядро не запускаю:\n{result.stdout[-1500:]}{result.stderr[-500:]}')

    digest = upload_code(args.user, args.code_slug, token)
    argv += ['--expect-code-hash', digest]
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    folder = STAGING / 'kernels' / slug
    build_kernel(folder, runner, args.user, slug, data_slug, args.code_slug, argv, bool(kaggle.get('internet', True)), data_owner=args.data_owner)
    active = push_and_track(kernel, digest, folder, out, timeout, token)
    if args.submit_only:
        print(f'Отправлено: https://www.kaggle.com/code/{kernel}\nТерминал можно закрыть. Обучение выполняется на Kaggle.', flush=True)
        return
    time.sleep(15)
    status = wait(kernel, args.poll, timeout, token)
    (active / slug).unlink(missing_ok=True)
    fetch(kernel, out, token)
    if status != 'complete':
        raise SystemExit(f'ядро завершилось со статусом {status}: лог в {out}')


if __name__ == '__main__':
    main()
