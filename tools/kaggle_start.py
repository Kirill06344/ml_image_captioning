"""Verify an explicit Kaggle account/quota and safely start or resume a GPU experiment.

Use from a terminal with network access. Full training requires a locally verified, successful GPU smoke.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import Corpus, write_json, sha256
from src.codehash import package_hash
from src.config import load_config


def check_account(api, expected):
    actual = api.get_config_value('username')
    if not actual or actual.casefold() != expected.casefold():
        raise ValueError(f'Ожидался аккаунт {expected}, активен {actual or "неизвестный"}. Ничего не отправлено.')
    return actual


def quota_summary(response, minimum):
    gpu = response.gpu_quota
    if gpu is None or gpu.time_used is None or gpu.total_time_allowed is None or gpu.time_reserved is None:
        raise ValueError('Не удалось проверить GPU-квоту и зарезервированное время; отправка остановлена.')
    total = gpu.total_time_allowed.total_seconds() / 3600
    used = gpu.time_used.total_seconds() / 3600
    reserved = gpu.time_reserved.total_seconds() / 3600
    available = max(0., total - used - reserved)
    if available < minimum:
        raise ValueError(f'Для запуска нужно {minimum:.2f} GPU-ч, доступно с учётом резерва {available:.2f}.')
    return {'total_hours': total, 'used_hours': used, 'reserved_hours': reserved, 'available_hours': available}


def http_status(error):
    """The Kaggle wrapper hides HTTP 403 inside a ValueError's exception context."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code = getattr(getattr(error, 'response', None), 'status_code', None)
        if code is not None:
            return code
        error = error.__cause__ or error.__context__
    return None


def has_submission_record(kernel):
    slug = kernel.split('/')[-1]
    return ((ROOT / 'runs/.kaggle/active' / slug).exists() or
            (ROOT / 'runs' / slug / 'submission.json').exists())


def absent_from_own_kernels(api, kernel):
    """403 alone proves nothing. Only a successful complete owner listing can establish absence."""
    owner = kernel.split('/')[0]
    actual = api.get_config_value('username')
    if not actual or owner.casefold() != actual.casefold():
        return False
    for page in range(1, 1001):
        rows = api.kernels_list(mine=True, page=page, page_size=100)
        if rows is None:
            return False
        if any(getattr(row, 'ref', '').casefold() == kernel.casefold() for row in rows):
            return False
        if not rows:
            return True
    return False


def kernel_state(api, kernel):
    try:
        response = api.kernels_status(kernel)
    except Exception as exc:
        code = http_status(exc)
        if not has_submission_record(kernel):
            if code == 404:
                return 'absent'
            if code == 403:
                try:
                    if absent_from_own_kernels(api, kernel):
                        print('Новое ядро отсутствует в полном списке собственных ядер:', kernel, flush=True)
                        return 'absent'
                except Exception:
                    pass
        detail = f'HTTP {code}' if code is not None else type(exc).__name__
        raise RuntimeError(f'Статус ядра {kernel} неизвестен ({detail}). Повторная отправка запрещена.') from exc
    value = getattr(response.status, 'name', str(response.status)).split('.')[-1].lower()
    if value not in ('queued', 'running', 'complete', 'error', 'cancel_acknowledged', 'cancel_requested'):
        raise ValueError('Kaggle вернул неизвестный статус; ничего не отправлено.')
    return value


def action_for(state):
    if state == 'absent':
        return 'start'
    if state in ('queued', 'running', 'cancel_requested'):
        return 'attach'
    if state in ('complete', 'error', 'cancel_acknowledged'):
        return 'fetch'
    raise ValueError('Неизвестный статус: не отправлять повторно')


def connect(token_file=None):
    # Scoped to this process and its children; never write/export/copy credentials.
    os.environ.pop('KAGGLE_API_TOKEN', None)
    if token_file is not None:
        path = Path(token_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError('Файл access token не найден; укажите существующий --token-file.')
        os.environ['KAGGLE_API_TOKEN'] = str(path)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def call(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / 'tools' / script), *arguments], cwd=ROOT, check=True)


def read_failure(stage, dataset, error):
    code = http_status(error)
    detail = f'HTTP {code}' if code is not None else type(error).__name__
    return RuntimeError(f'{stage}: {dataset} ({detail}). Чтение не подтверждено; ничего не отправлено.')


def check_remote_data(api, dataset, corpus):
    """Prove Viewer access by downloading data, not by querying owner-only processing APIs."""
    owner = dataset.split('/')[0]
    actor = api.get_config_value('username')
    is_owner = bool(actor and actor.casefold() == owner.casefold())
    with tempfile.TemporaryDirectory(prefix='caption-kaggle-check-') as tmp:
        # Privacy management belongs to the owner. A Viewer need only be able to read the data;
        # this code never changes sharing or visibility.
        if is_owner:
            try:
                meta_path = api.dataset_metadata(dataset, tmp)
                metadata = json.loads(Path(meta_path).read_text())
            except Exception as exc:
                if http_status(exc) == 404:
                    return False
                raise read_failure('Не удалось прочитать метаданные владельца', dataset, exc) from exc
            metadata = metadata.get('info', metadata)
            if metadata.get('isPrivate') is not True:
                raise ValueError('Существующий датасет не подтверждён как приватный; ничего не изменено.')
        try:
            api.dataset_download_file(dataset, 'manifest.json', path=tmp, quiet=True, force=True, unzip=True)
            manifest = json.loads((Path(tmp) / 'manifest.json').read_text())
        except Exception as exc:
            raise read_failure('Не удалось скачать файл manifest.json', dataset, exc) from exc
        if manifest != corpus.manifest:
            raise ValueError('Manifest на Kaggle отличается от локального. Версия автоматически не меняется.')
    print('Чтение датасета и совпадение manifest подтверждены:', dataset, flush=True)
    return True


def diagnose(api, kernel):
    """Read server-side state and metadata. Never push, cancel or modify datasets/code."""
    report = {'kernel': kernel, 'read_only': True}
    try:
        result = api.kernels_status(kernel)
        report['status'] = getattr(result.status, 'name', str(result.status)).split('.')[-1].lower()
        report['failure_message'] = result.failure_message or ''
    except Exception as exc:
        report['status_error'] = {'type': type(exc).__name__, 'http_status': http_status(exc)}
    try:
        response = api.quota_view()
        report['quota'] = quota_summary(response, 0)
    except Exception as exc:
        report['quota_error'] = {'type': type(exc).__name__, 'http_status': http_status(exc)}
    try:
        with tempfile.TemporaryDirectory(prefix='caption-server-metadata-') as tmp:
            api.kernels_pull(kernel, path=tmp, metadata=True, quiet=True)
            metadata = json.loads((Path(tmp) / 'kernel-metadata.json').read_text())
        fields = ['id', 'enable_gpu', 'enable_tpu', 'enable_internet', 'machine_shape', 'dataset_sources', 'docker_image']
        report['server_metadata'] = {key: metadata.get(key) for key in fields}
    except Exception as exc:
        report['metadata_error'] = {'type': type(exc).__name__, 'http_status': http_status(exc)}
    path = ROOT / 'runs/diagnostics' / (kernel.replace('/', '--') + '.json')
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('Диагностика сохранена:', path)
    return report


def require_verified_gpu_smoke(corpus):
    """Bind full training to the audited real GPU smoke, current source and exact data."""
    path = ROOT / 'docs/validation/gpu_smoke.json'
    if not path.is_file():
        raise ValueError('Перед полным обучением требуется проверенный GPU smoke: docs/validation/gpu_smoke.json')
    report = json.loads(path.read_text())
    if report.get('status') != 'passed' or report.get('dataset_fingerprint') != corpus.manifest['fingerprint']:
        raise ValueError('GPU smoke не подтверждён для этой выборки')
    if report.get('code_hash') != package_hash(ROOT / 'src'):
        raise ValueError('Код после GPU smoke изменился; сначала проверить новый smoke')
    root = ROOT / 'runs' / report['run']
    run = json.loads((root / 'run.json').read_text())
    if not run.get('completed') or run.get('device') != 'cuda' or not run.get('checkpoint_reloaded'):
        raise ValueError('Артефакты GPU smoke не подтверждают успешное завершение')
    if sha256(root / 'best.pt') != report['checkpoint_sha256']:
        raise ValueError('Checkpoint проверенного GPU smoke изменился')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user', required=True)
    p.add_argument('--config', type=Path, default=Path('configs/experiments/smoke.yaml'), help='конфигурация smoke или полного опыта')
    p.add_argument('--token-file', type=Path, help='отдельный файл access token исполнителя; глобальная авторизация не меняется')
    p.add_argument('--data-owner', default=None, help='владелец общего COCO; чужие данные не загружаются и не обновляются')
    p.add_argument('--min-gpu-hours', type=float, default=None)
    p.add_argument('--submit-only', action='store_true', help='отправить и выйти; существующий запуск повторно не отправляется')
    p.add_argument('--diagnose', action='store_true', help='прочитать статус, сообщение сервера, квоту и реальные метаданные; ничего не запускать')
    p.add_argument('--check-only', action='store_true', help='account/quota/status/shared manifest only; no upload or launch')
    args = p.parse_args(argv)
    if args.min_gpu_hours is not None and args.min_gpu_hours <= 0:
        p.error('min-gpu-hours должен быть положительным')
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    is_smoke = config['train'].get('smoke_images', 0) > 0
    if is_smoke and (config['train']['smoke_images'] < 2 or config['train'].get('max_steps', 0) < 1):
        raise ValueError('Smoke требует хотя бы два изображения и ограничение max_steps')
    minimum_hours = args.min_gpu_hours if args.min_gpu_hours is not None else (.25 if is_smoke else 2.0)
    slug = config['slug']
    api = connect(args.token_file) if args.token_file else connect()
    check_account(api, args.user)
    kernel = args.user + '/' + slug
    if args.diagnose:
        diagnose(api, kernel)
        return
    state = kernel_state(api, kernel)
    print('Подтверждён аккаунт:', args.user, 'ядро:', state, flush=True)
    action = action_for(state)
    data_owner = args.data_owner or args.user
    command = ('--config', str(config_path), '--user', args.user, '--data-owner', data_owner)
    if args.token_file:
        command += ('--token-file', str(args.token_file.expanduser().resolve()))
    if args.submit_only:
        command += ('--submit-only',)
    if action != 'start' and not args.check_only:
        if args.submit_only:
            print(f'Существующий запуск: https://www.kaggle.com/code/{kernel}, статус {state}. Нового запуска нет.', flush=True)
            if state in ('error', 'cancel_acknowledged'):
                raise ValueError('Запуск завершился без успеха; получить результаты через --fetch-only и разобрать лог.')
            return
        # No quota is needed to fetch an existing result; no code/data mutation on resume.
        call('kaggle_train.py', *command, '--attach' if action == 'attach' else '--fetch-only')
        if state in ('error', 'cancel_acknowledged'):
            raise ValueError('Получены файлы неуспешного запуска. Новый запуск не отправлен; требуется разбор лога.')
        return
    quota = quota_summary(api.quota_view(), minimum_hours)
    print('GPU-квота:', json.dumps(quota, ensure_ascii=False), flush=True)
    write_json(ROOT / 'runs/.kaggle' / f'preflight-{args.user}.json',
               {'user': args.user, 'kernel': kernel, 'kernel_state': state, 'quota': quota})
    if args.check_only:
        corpus = Corpus(ROOT / 'data/processed')
        dataset = data_owner + '/coco2014-caption-5k'
        if not check_remote_data(api, dataset, corpus):
            raise ValueError(f'Датасет {dataset} недоступен; проверьте загрузку и доступ исполнителя.')
        print('Доступ к датасету и совпадение manifest подтверждены:', dataset)
        print('Только проверка: загрузок и запуска не было.')
        return
    corpus = Corpus(ROOT / 'data/processed')
    if not is_smoke:
        require_verified_gpu_smoke(corpus)
    dataset = data_owner + '/coco2014-caption-5k'
    shared_verified = False
    if data_owner.casefold() != args.user.casefold():
        shared_verified = check_remote_data(api, dataset, corpus)
        if not shared_verified:
            raise ValueError(f'Общий датасет {dataset} недоступен. Проверьте Viewer-доступ.')
    # Run all local checks before first external mutation.
    subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
    corpus = Corpus(ROOT / 'data/processed')
    smoke = json.loads((ROOT / 'runs/local-coco-smoke/run.json').read_text())
    if not smoke.get('completed') or smoke['dataset_fingerprint'] != corpus.manifest['fingerprint']:
        raise ValueError('Требуется успешный локальный smoke на этой же выборке')
    active = ROOT / 'runs/.kaggle/active' / slug
    if active.exists():
        raise ValueError('Есть локальная запись отправленного smoke при удалённом 404; сначала выяснить статус.')
    dataset = data_owner + '/coco2014-caption-5k'
    if not shared_verified and not check_remote_data(api, dataset, corpus):
        if data_owner.casefold() != args.user.casefold():
            raise ValueError(f'Общий датасет {dataset} недоступен. Сначала загрузить его владельцем и выдать доступ исполнителю.')
        call('kaggle_upload.py', '--dir', str(corpus.root), '--slug', 'coco2014-caption-5k', '--user', args.user)
        for _ in range(30):
            if api.dataset_status(dataset) == 'ready':
                break
            print('Ожидание обработки датасета COCO...', flush=True)
            time.sleep(20)
        else:
            raise ValueError('Датасет ещё обрабатывается. Запустите ту же команду позже; версия не изменялась.')
        if not check_remote_data(api, dataset, corpus):
            raise ValueError('Загруженный датасет не найден; ядро не отправлено.')
    # Recheck quota/status after any lengthy upload, immediately before submitting code/kernel.
    quota_summary(api.quota_view(), minimum_hours)
    if kernel_state(api, kernel) != 'absent':
        raise ValueError('Статус smoke изменился после проверки; повторно запустите команду для attach/fetch.')
    call('kaggle_train.py', *command)


if __name__ == '__main__':
    try:
        main()
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        # Authentication/network exceptions can contain request details. Never echo them or a traceback.
        if isinstance(exc, (ValueError, RuntimeError)):
            print(str(exc), file=sys.stderr)
        else:
            print(f'Проверка/запуск остановлены ({type(exc).__name__}). Проверьте сеть и вход в указанный Kaggle-аккаунт; токены не присылайте.', file=sys.stderr)
        raise SystemExit(1)
