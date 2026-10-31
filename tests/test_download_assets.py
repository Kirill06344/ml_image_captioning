"""Download integrity and deterministic selection tests without network access."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import download_assets as assets
from tools.prepare_data import stable_ids


class Response(io.BytesIO):
    def __init__(self, value, size=None):
        super().__init__(value)
        self.headers = {'Content-Length': str(len(value) if size is None else size)}


def test_atomic_download_reuses_only_verified_content(tmp_path, monkeypatch):
    payload = b'valid source bytes'
    calls = []

    def open_source(request, timeout):
        calls.append(request.full_url)
        return Response(payload)

    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_source)
    target = tmp_path / 'source.bin'
    digest = assets.download('https://example.invalid/source', target)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert not target.with_name('source.bin.part').exists()
    assert json.loads(target.with_name('source.bin.download.json').read_text())['sha256'] == digest
    assets.download('https://example.invalid/source', target)
    assert len(calls) == 1
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='отличается'):
        assets.download('https://example.invalid/source', target)


def test_incomplete_transfer_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **kw: Response(b'truncated', size=1000))
    monkeypatch.setattr(assets.time, 'sleep', lambda _: None)
    target = tmp_path / 'source.bin'
    with pytest.raises(RuntimeError, match='Content-Length'):
        assets.download('https://example.invalid/source', target)
    assert not target.exists()
    assert not target.with_name('source.bin.part').exists()


def test_weights_hash_is_checked_before_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **kw: Response(b'wrong weights'))
    monkeypatch.setattr(assets.time, 'sleep', lambda _: None)
    target = tmp_path / 'weights.pth'
    with pytest.raises(RuntimeError, match='официальным'):
        assets.download('https://example.invalid/weights', target, expected_prefix='f37072fd')
    assert not target.exists()


def test_selection_matches_preparation_and_rejects_unsafe_names(tmp_path):
    paths = []
    for split, keys in [('train', [1, 2, 3]), ('val', [4, 5, 6, 7])]:
        raw = {'images': [{'id': key, 'file_name': f'{key}.jpg'} for key in keys],
               'annotations': [{'image_id': key, 'caption': 'a photo'} for key in keys]}
        path = tmp_path / f'{split}.json'
        path.write_text(json.dumps(raw))
        paths.append(path)
    planned = assets.selected_images(*paths, 2, 1, 1)
    expected = [('train2014', f'{key}.jpg') for key in stable_ids([1, 2, 3], 2)]
    expected += [('val2014', f'{key}.jpg') for key in stable_ids([4, 5, 6, 7], 2)]
    assert planned == expected
    raw = json.loads(paths[0].read_text())
    for image in raw['images']:
        image['file_name'] = '../unsafe.jpg'
    paths[0].write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='Небезопасное'):
        assets.selected_images(*paths, 2, 1, 1)


def test_plan_has_no_network_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('network in plan'))
    assets.main(['--raw', str(tmp_path / 'raw'), '--out', str(tmp_path / 'out'), '--plan'])
    assert '4000/500/500' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
