from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from tools.kaggle_start import action_for, check_account, kernel_state, quota_summary


def test_account_must_match_explicit_user():
    api = NS(get_config_value=lambda _: 'another-user')
    with pytest.raises(ValueError, match='Ожидался аккаунт uliabzh'):
        check_account(api, 'uliabzh')
    api.get_config_value = lambda _: 'uliabzh'
    assert check_account(api, 'uliabzh') == 'uliabzh'


def test_quota_deducts_both_usage_and_reservations():
    quota = NS(time_used=timedelta(hours=20), total_time_allowed=timedelta(hours=30),
               time_reserved=timedelta(hours=8))
    assert quota_summary(NS(gpu_quota=quota), 1)['available_hours'] == 2
    with pytest.raises(ValueError, match='доступно'):
        quota_summary(NS(gpu_quota=quota), 3)
    quota.time_reserved = None
    with pytest.raises(ValueError, match='зарезервированное'):
        quota_summary(NS(gpu_quota=quota), 1)


def test_network_or_forbidden_never_means_absent():
    for code in [None, 403, 500]:
        def fail(_):
            error = OSError('unavailable')
            error.response = NS(status_code=code)
            raise error
        with pytest.raises(RuntimeError, match='Повторная отправка запрещена'):
            kernel_state(NS(kernels_status=fail), 'uliabzh/caption-smoke')


def test_only_explicit_404_allows_first_start():
    def missing(_):
        error = OSError('not found')
        error.response = NS(status_code=404)
        raise error
    assert action_for(kernel_state(NS(kernels_status=missing), 'x/y')) == 'start'
    for state in ['running', 'queued']:
        assert action_for(state) == 'attach'
    for state in ['complete', 'error', 'cancel_acknowledged']:
        assert action_for(state) == 'fetch'
    with pytest.raises(ValueError):
        action_for('unknown')


def test_check_only_validates_shared_manifest_without_launch(tmp_path, monkeypatch):
    from tools import kaggle_start as start
    api = NS(get_config_value=lambda _: 'uliabzh', quota_view=lambda: NS(gpu_quota=NS(
        time_used=timedelta(hours=0), time_reserved=timedelta(hours=0), total_time_allowed=timedelta(hours=10))))
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    monkeypatch.setattr(start, 'connect', lambda: api)
    monkeypatch.setattr(start, 'kernel_state', lambda *a: 'absent')
    monkeypatch.setattr(start, 'Corpus', lambda _: NS())
    checked = []
    monkeypatch.setattr(start, 'check_remote_data', lambda api, dataset, corpus: checked.append(dataset) or True)
    monkeypatch.setattr(start, 'call', lambda *a: pytest.fail('launch in check-only'))
    from pathlib import Path
    config = Path(__file__).resolve().parents[1] / 'configs/experiments/smoke.yaml'
    start.main(['--user', 'uliabzh', '--data-owner', 'badzhik', '--config', str(config), '--check-only'])
    assert checked == ['badzhik/coco2014-caption-5k']


def test_explicit_token_path_is_forwarded_on_resume(tmp_path, monkeypatch):
    from tools import kaggle_start as start
    token = tmp_path / 'existing-access-token'
    token.touch()
    connected = []
    def connect(path):
        connected.append(path)
        return NS(get_config_value=lambda _: 'uliabzh')
    monkeypatch.setattr(start, 'connect', connect)
    monkeypatch.setattr(start, 'kernel_state', lambda *a: 'running')
    commands = []
    monkeypatch.setattr(start, 'call', lambda *a: commands.append(a))
    start.main(['--user', 'uliabzh', '--data-owner', 'badzhik', '--token-file', str(token)])
    assert connected == [token]
    assert len(commands) == 1 and '--attach' in commands[0]
    assert commands[0][commands[0].index('--token-file') + 1] == str(token)


def test_nonexistent_explicit_token_is_rejected_before_import(tmp_path, monkeypatch):
    from tools.kaggle_start import connect
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'ignored')
    with pytest.raises(ValueError, match='Файл access token не найден'):
        connect(tmp_path / 'missing')


def wrapped_http_error(code):
    inner = OSError('server response')
    inner.response = NS(status_code=code)
    outer = ValueError('Kaggle wrapper message')
    outer.__context__ = inner
    return outer


def test_new_kernel_403_needs_complete_owner_listing(tmp_path, monkeypatch):
    from tools import kaggle_start as start
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    def fail(_):
        raise wrapped_http_error(403)
    pages = []
    def listing(**kw):
        pages.append(kw['page'])
        return [NS(ref='uliabzh/another')] if kw['page'] == 1 else []
    api = NS(kernels_status=fail, get_config_value=lambda _: 'uliabzh', kernels_list=listing)
    assert start.kernel_state(api, 'uliabzh/caption-smoke') == 'absent'
    assert pages == [1, 2]
    api.kernels_list = lambda **kw: [NS(ref='uliabzh/caption-smoke')]
    with pytest.raises(RuntimeError, match='HTTP 403'):
        start.kernel_state(api, 'uliabzh/caption-smoke')


def test_submission_journal_prevents_retry_despite_empty_remote_list(tmp_path, monkeypatch):
    from tools import kaggle_start as start
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    marker = tmp_path / 'runs/caption-smoke/submission.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    for code in [403, 404]:
        def fail(_):
            raise wrapped_http_error(code)
        api = NS(kernels_status=fail, get_config_value=lambda _: 'uliabzh', kernels_list=lambda **kw: [])
        with pytest.raises(RuntimeError, match='Повторная отправка запрещена'):
            start.kernel_state(api, 'uliabzh/caption-smoke')


def test_failed_listing_or_wrong_owner_cannot_authorize_push(tmp_path, monkeypatch):
    from tools import kaggle_start as start
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    def fail(_):
        raise wrapped_http_error(403)
    for account, listing in [('badzhik', lambda **kw: []), ('uliabzh', lambda **kw: None)]:
        api = NS(kernels_status=fail, get_config_value=lambda _: account, kernels_list=listing)
        with pytest.raises(RuntimeError, match='HTTP 403'):
            start.kernel_state(api, 'uliabzh/caption-smoke')


def remote_data_api(tmp_path, read_allowed=True):
    import json
    manifest = {'fingerprint': 'verified-fixture', 'splits': {}}
    def status(_):
        raise wrapped_http_error(403)
    def metadata(dataset, path):
        if not read_allowed:
            raise wrapped_http_error(403)
        target = tmp_path / 'private-metadata.json'
        target.write_text(json.dumps({'isPrivate': True}))
        return str(target)
    def download(dataset, name, path, **kwargs):
        from pathlib import Path
        (Path(path) / name).write_text(json.dumps(manifest))
    api = NS(get_config_value=lambda _: 'uliabzh', dataset_status=status, dataset_metadata=metadata, dataset_download_file=download)
    return api, NS(manifest=manifest)


def test_shared_dataset_status_403_requires_successful_private_read(tmp_path):
    from tools.kaggle_start import check_remote_data
    api, corpus = remote_data_api(tmp_path)
    assert check_remote_data(api, 'badzhik/coco2014-caption-5k', corpus)
    corpus.manifest = {'fingerprint': 'different'}
    with pytest.raises(ValueError, match='Manifest на Kaggle отличается'):
        check_remote_data(api, 'badzhik/coco2014-caption-5k', corpus)



def test_shared_dataset_read_denied_is_not_bypassed(tmp_path):
    from tools.kaggle_start import check_remote_data
    api, corpus = remote_data_api(tmp_path)
    def denied(*a, **kw):
        raise wrapped_http_error(403)
    api.dataset_download_file = denied
    with pytest.raises(RuntimeError, match='manifest.json.*HTTP 403'):
        check_remote_data(api, 'badzhik/coco2014-caption-5k', corpus)


def test_shared_dataset_network_failure_stops_on_actual_file_read():
    from tools.kaggle_start import check_remote_data
    def fail(*a, **kw):
        raise OSError('network unavailable')
    api = NS(get_config_value=lambda _: 'uliabzh', dataset_download_file=fail)
    with pytest.raises(RuntimeError, match='Не удалось скачать файл manifest.json'):
        check_remote_data(api, 'badzhik/coco2014-caption-5k', NS())


@pytest.mark.parametrize('status_code', [403, 404])
def test_viewer_can_read_manifest_when_processing_and_admin_endpoints_are_unavailable(tmp_path, status_code):
    from tools.kaggle_start import check_remote_data
    api, corpus = remote_data_api(tmp_path)
    def owner_endpoint_unavailable(*a):
        raise wrapped_http_error(status_code)
    api.dataset_status = owner_endpoint_unavailable
    api.dataset_metadata = owner_endpoint_unavailable
    assert check_remote_data(api, 'badzhik/coco2014-caption-5k', corpus)


def test_owner_still_requires_private_metadata(tmp_path):
    from tools.kaggle_start import check_remote_data
    api, corpus = remote_data_api(tmp_path, read_allowed=False)
    api.get_config_value = lambda _: 'badzhik'
    with pytest.raises(RuntimeError, match='метаданные владельца.*HTTP 403'):
        check_remote_data(api, 'badzhik/coco2014-caption-5k', corpus)


def test_main_account_smoke_uses_separate_slug_and_config(monkeypatch):
    from tools import kaggle_start as start
    monkeypatch.setattr(start, 'connect', lambda: NS(get_config_value=lambda _: 'badzhik'))
    checked = []
    def state(api, kernel):
        checked.append(kernel)
        return 'running'
    monkeypatch.setattr(start, 'kernel_state', state)
    calls = []
    monkeypatch.setattr(start, 'call', lambda *args: calls.append(args))
    start.main(['--user', 'badzhik', '--data-owner', 'badzhik', '--config', 'configs/experiments/smoke-main.yaml'])
    assert checked == ['badzhik/caption-smoke-main']
    assert '--attach' in calls[0]
    config_path = calls[0][calls[0].index('--config') + 1]
    assert config_path.endswith('configs/experiments/smoke-main.yaml')


def test_diagnose_reads_server_settings_without_submission(tmp_path, monkeypatch):
    import json
    from tools import kaggle_start as start
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    def pull(kernel, path, metadata, quiet):
        from pathlib import Path
        (Path(path) / 'kernel-metadata.json').write_text(json.dumps({
            'id': kernel, 'enable_gpu': True, 'enable_tpu': False, 'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': ['badzhik/coco2014-caption-5k', 'badzhik/caption-code']}))
    api = NS(kernels_status=lambda _: NS(status=NS(name='QUEUED'), failure_message=''),
             quota_view=lambda: NS(gpu_quota=NS(time_used=timedelta(),time_reserved=timedelta(),total_time_allowed=timedelta(hours=30))),
             kernels_pull=pull)
    report = start.diagnose(api, 'badzhik/caption-smoke-main')
    assert report['status'] == 'queued'
    assert report['server_metadata']['enable_gpu'] is True
    assert report['server_metadata']['machine_shape'] == 'NvidiaTeslaT4'
    assert report['quota']['available_hours'] == 30
    assert report['read_only']
    assert not (tmp_path / 'runs/.kaggle/active').exists()
