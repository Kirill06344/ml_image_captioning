import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kaggle_train', ROOT / 'tools' / 'kaggle_train.py')
kaggle_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kaggle_train)


@pytest.mark.parametrize('output, expected', [
    ('badzhik/x has status "KernelWorkerStatus.RUNNING"', 'running'),
    ("badzhik/x has status 'complete'", 'complete'),
    ('HTTPSConnectionPool: NameResolutionError', None),   # сетевой сбой — не падение ядра
    ('', None),
])
def test_worker_status(output, expected):
    assert kaggle_train.worker_status(output) == expected


@pytest.mark.parametrize('status, expected', [('complete', True), ('error', True), ('cancel_acknowledged', True),
                                              ('running', False), ('queued', False), (None, False)])
def test_finished(status, expected):
    assert kaggle_train.finished(status) is expected


def test_token_only_for_given_call(monkeypatch):
    captured = {}

    def fake(command, capture_output, text, env):
        captured.update(env)
        return subprocess.CompletedProcess(command, 0, 'ok', '')

    monkeypatch.setenv('KAGGLE_API_TOKEN', 'leaked')
    monkeypatch.setattr(kaggle_train.subprocess, 'run', fake)
    kaggle_train.run('kernels', 'status', 'x')
    assert 'KAGGLE_API_TOKEN' not in captured
    captured.clear()
    kaggle_train.run('kernels', 'status', 'x', token='/path/token')
    assert captured['KAGGLE_API_TOKEN'] == '/path/token'


def test_generated_kernel_is_private_and_calls_trainer(tmp_path):
    import json

    kaggle_train.build_kernel(tmp_path, 'runner', 'owner', 'run', 'data', 'code', ['--data', 'x'], internet=True)
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True
    assert metadata['id'] == 'runner/run'   # ядро и квота — у исполнителя, код и данные — у владельца
    assert metadata['dataset_sources'] == ['owner/data', 'owner/code']
    source = (tmp_path / 'run_training.py').read_text()
    compile(source, 'run_training.py', 'exec')
    assert "main(['--data', 'x'])" in source


def test_shared_data_owner_is_independent_of_code_owner(tmp_path):
    import json
    kaggle_train.build_kernel(tmp_path, 'uliabzh', 'uliabzh', 'caption-smoke',
                              'coco2014-caption-5k', 'caption-code', [], internet=True, data_owner='badzhik')
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'uliabzh/caption-smoke'
    assert metadata['dataset_sources'] == ['badzhik/coco2014-caption-5k', 'uliabzh/caption-code']
    assert metadata['is_private']


def test_explicit_token_file_reaches_all_owner_cli_calls(monkeypatch):
    captured = []
    def fake(command, capture_output, text, env):
        captured.append(env.get('KAGGLE_API_TOKEN'))
        return subprocess.CompletedProcess(command, 0, 'ok', '')
    monkeypatch.setattr(kaggle_train, 'AUTH_TOKEN_FILE', '/existing/uliabzh-access-token')
    monkeypatch.setattr(kaggle_train.subprocess, 'run', fake)
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'unrelated-secret')
    kaggle_train.run('datasets', 'list')
    kaggle_train.run('kernels', 'status', 'uliabzh/caption-smoke')
    kaggle_train.run('kernels', 'status', 'other/x', token='/existing/other-token')
    assert captured == ['/existing/uliabzh-access-token', '/existing/uliabzh-access-token', '/existing/other-token']
