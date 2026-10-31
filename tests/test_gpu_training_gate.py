import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.data import sha256, write_json
from src.codehash import package_hash
from tools import kaggle_start as start


def fixture_smoke(root):
    (root / 'src').mkdir()
    (root / 'src/train.py').write_text('fixture source')
    checkpoint = root / 'runs/smoke/best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture checkpoint')
    write_json(checkpoint.parent / 'run.json', {'completed': True, 'device': 'cuda', 'checkpoint_reloaded': True})
    report = {'status':'passed','run':'smoke','dataset_fingerprint':'same-data',
              'code_hash': package_hash(root / 'src'), 'checkpoint_sha256':sha256(checkpoint)}
    write_json(root / 'docs/validation/gpu_smoke.json', report)
    return checkpoint


def test_full_training_requires_successful_gpu_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    corpus = NS(manifest={'fingerprint':'same-data'})
    with pytest.raises(ValueError, match='проверенный GPU smoke'):
        start.require_verified_gpu_smoke(corpus)
    checkpoint = fixture_smoke(tmp_path)
    assert start.require_verified_gpu_smoke(corpus)['status'] == 'passed'
    checkpoint.write_bytes(b'altered')
    with pytest.raises(ValueError, match='Checkpoint'):
        start.require_verified_gpu_smoke(corpus)


def test_changed_source_or_dataset_requires_new_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    fixture_smoke(tmp_path)
    with pytest.raises(ValueError, match='выборки'):
        start.require_verified_gpu_smoke(NS(manifest={'fingerprint':'different-data'}))
    (tmp_path / 'src/train.py').write_text('new source')
    with pytest.raises(ValueError, match='Код после GPU smoke изменился'):
        start.require_verified_gpu_smoke(NS(manifest={'fingerprint':'same-data'}))
