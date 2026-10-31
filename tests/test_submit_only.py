import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

from tools import kaggle_train as train
from tools import kaggle_start as start


def test_submit_only_exits_after_ack_without_wait_or_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    monkeypatch.setattr(train, 'STAGING', tmp_path / 'staging')
    monkeypatch.setattr(train, 'load_config', lambda _: {'slug': 'caption-control', 'kaggle': {
        'data_slug': 'coco2014-caption-5k', 'timeout_minutes': 120}, 'train': {'model': 'resnet18-transformer'}})
    monkeypatch.setattr(train, 'upload_code', lambda *a: 'verified-code-hash')
    monkeypatch.setattr(train, 'wait', lambda *a, **kw: pytest.fail('submit-only must not wait'))
    monkeypatch.setattr(train, 'fetch', lambda *a, **kw: pytest.fail('submit-only must not fetch'))
    monkeypatch.setattr(train.time, 'sleep', lambda _: pytest.fail('submit-only must not sleep after push'))
    def push(*args, **kwargs):
        assert args[:2] == ('kernels', 'push')
        assert args[args.index('--timeout') + 1] == '7200'
        record = json.loads((tmp_path / 'runs/caption-control/submission.json').read_text())
        assert record['state'] == 'push_attempted'
        return 'Kernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/badzhik/caption-control'
    monkeypatch.setattr(train, 'run', push)
    monkeypatch.setattr(sys, 'argv', ['kaggle_train.py', '--config', 'config.yaml', '--user', 'badzhik', '--submit-only', '--skip-tests'])
    train.main()
    receipt = json.loads((tmp_path / 'runs/caption-control/submission.json').read_text())
    assert receipt['state'] == 'submitted' and receipt['submitted_at_utc']
    assert (tmp_path / 'staging/active/caption-control').exists()


def test_missing_ack_keeps_attempt_marker_and_does_not_claim_success(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'STAGING', tmp_path / 'staging')
    monkeypatch.setattr(train, 'run', lambda *a, **kw: 'Kernel push error: permission denied')
    out = tmp_path / 'caption-control'
    out.mkdir()
    with pytest.raises(SystemExit, match='не подтвердил отправку'):
        train.push_and_track('badzhik/caption-control', 'digest', tmp_path, out, 120)
    assert json.loads((out / 'submission.json').read_text())['state'] == 'push_attempted'
    assert (tmp_path / 'staging/active/caption-control').exists()


@pytest.mark.parametrize('state', ['queued', 'running', 'complete'])
def test_submit_only_does_not_wait_or_resubmit_existing_job(monkeypatch, state):
    monkeypatch.setattr(start, 'connect', lambda: NS(get_config_value=lambda _: 'badzhik'))
    monkeypatch.setattr(start, 'kernel_state', lambda *a: state)
    monkeypatch.setattr(start, 'call', lambda *a: pytest.fail('must not launch or attach existing job'))
    start.main(['--user', 'badzhik', '--config', 'configs/experiments/control.yaml', '--submit-only'])
