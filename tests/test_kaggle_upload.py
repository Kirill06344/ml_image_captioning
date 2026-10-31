from types import SimpleNamespace as NS

import pytest
from tools import kaggle_upload


def test_wrong_account_prevents_metadata_and_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_upload, 'connect', lambda: NS(get_config_value=lambda _: 'uliabzh'))
    monkeypatch.setattr(kaggle_upload, 'run', lambda *a, **kw: pytest.fail('upload on wrong account'))
    with pytest.raises(ValueError, match='Ожидался аккаунт badzhik'):
        kaggle_upload.main(['--dir', str(tmp_path), '--slug', 'coco2014-caption-5k', '--user', 'badzhik'])
    assert not (tmp_path / 'dataset-metadata.json').exists()


def test_matching_existing_dataset_is_not_uploaded_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_upload, 'connect', lambda: NS(get_config_value=lambda _: 'badzhik'))
    monkeypatch.setattr(kaggle_upload, 'Corpus', lambda _: NS())
    monkeypatch.setattr(kaggle_upload, 'owned_dataset_exists', lambda *a: True)
    monkeypatch.setattr(kaggle_upload, 'check_remote_data', lambda *a: True)
    monkeypatch.setattr(kaggle_upload, 'run', lambda *a, **kw: pytest.fail('duplicate upload'))
    kaggle_upload.main(['--dir', str(tmp_path), '--slug', 'coco2014-caption-5k', '--user', 'badzhik'])
    assert not (tmp_path / 'dataset-metadata.json').exists()


def test_new_dataset_is_created_without_status_query(tmp_path, monkeypatch):
    api = NS(get_config_value=lambda _: 'badzhik', dataset_list=lambda **kw: [])
    monkeypatch.setattr(kaggle_upload, 'connect', lambda: api)
    monkeypatch.setattr(kaggle_upload, 'Corpus', lambda _: NS())
    monkeypatch.setattr(kaggle_upload, 'check_remote_data', lambda *a: pytest.fail('status for absent dataset'))
    monkeypatch.setattr(kaggle_upload, 'verify_private', lambda _: None)
    commands = []
    monkeypatch.setattr(kaggle_upload, 'run', lambda *a, **kw: commands.append(a) or 'Your private Dataset is being created')
    kaggle_upload.main(['--dir', str(tmp_path), '--slug', 'coco2014-caption-5k', '--user', 'badzhik'])
    assert len(commands) == 1
    assert commands[0][:2] == ('datasets', 'create')
    assert '--keep-tabular' in commands[0] and '--public' not in commands[0]


def test_owner_listing_checks_all_pages():
    seen = []
    def listing(**kw):
        seen.append(kw)
        return [NS(ref='badzhik/another')] if kw['page'] == 1 else [NS(ref='badzhik/coco2014-caption-5k')]
    assert kaggle_upload.owned_dataset_exists(NS(dataset_list=listing), 'badzhik/coco2014-caption-5k')
    assert seen == [{'mine': True, 'page': 1}, {'mine': True, 'page': 2}]


def test_failed_owner_listing_cannot_authorize_creation():
    def fail(**kw):
        error = OSError('failed request')
        error.response = NS(status_code=403)
        raise error
    with pytest.raises(RuntimeError, match='HTTP 403'):
        kaggle_upload.owned_dataset_exists(NS(dataset_list=fail), 'badzhik/coco2014-caption-5k')
