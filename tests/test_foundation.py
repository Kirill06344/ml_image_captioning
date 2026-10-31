import json
from pathlib import Path

import pytest

from src.config import load_config, train_argv
from src.train import build_parser
from tools.prepare_data import prepare
from tools.collect_results import check_comparable, collect


def test_caption_config_roundtrip():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs/experiments/smoke.yaml')
    args = build_parser().parse_args(['--data', 'input', '--out', 'output', *train_argv(build_parser(), config['train'])])
    assert args.max_steps == 10 and args.model == 'resnet18-transformer'
    assert config['slug'] == 'caption-smoke'


def fixture_annotations(root, name, ids):
    records = {'images': [], 'annotations': []}
    folder = root / name
    folder.mkdir()
    for key in ids:
        filename = f'image-{key}.jpg'
        (folder / filename).write_bytes(b'local fixture: file transfer only')
        records['images'].append({'id': key, 'file_name': filename})
        for j in range(2):
            records['annotations'].append({'image_id': key, 'caption': f'caption {key}-{j}'})
    path = root / (name + '.json')
    path.write_text(json.dumps(records))
    return path


def test_prepare_keeps_references_and_images_disjoint(tmp_path):
    train = fixture_annotations(tmp_path, 'train2014', [1, 2, 3])
    val = fixture_annotations(tmp_path, 'val2014', [4, 5, 6])
    out = tmp_path / 'prepared'
    manifest = prepare(train, val, tmp_path, out, 2, 1, 1)
    assert manifest['total_images'] == 4
    all_ids = []
    for split in ['train', 'val', 'test']:
        rows = [json.loads(line) for line in (out / f'{split}.jsonl').read_text().splitlines()]
        all_ids.extend(r['image_id'] for r in rows)
        assert all(len(r['references']) == 2 and (out / r['image']).is_file() for r in rows)
    assert len(set(all_ids)) == 4
    with pytest.raises(ValueError, match='не пуст'):
        prepare(train, val, tmp_path, out, 2, 1, 1)


def report(name, score, *, split='val', units='raw'):
    return {'run': name, 'role': 'candidate', 'split': split, 'n_images': 2,
            'dataset_fingerprint': 'fixture', 'image_ids_sha256': 'fixture-ids-' + split,
            'metric_units': {'CIDEr': units}, 'scorer': {'name': 'fixture'}, 'metrics': {'CIDEr': score}}


def test_comparison_rejects_other_metric_scale():
    with pytest.raises(ValueError, match='metric_units'):
        check_comparable([report('a', 1), report('b', 100, units='x100')])


def test_test_cannot_select_a_losing_model(tmp_path):
    runs = tmp_path / 'runs'
    for name, score in [('a', 1), ('b', 2)]:
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / 'scores_val.json').write_text(json.dumps(report(name, score)))
    _, selected = collect(runs, 'val', 'CIDEr', tmp_path / 'output')
    assert selected['final'] == 'b'
    (runs / 'a/scores_test.json').write_text(json.dumps(report('a', 2, split='test')))
    with pytest.raises(ValueError, match='Test разрешён'):
        collect(runs, 'test', 'CIDEr', tmp_path / 'output')


def test_single_full_run_has_bounded_runtime_and_no_smoke_limits():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs/experiments/control.yaml')
    args = build_parser().parse_args(['--data', 'data', '--out', 'out', *train_argv(build_parser(), config['train'])])
    assert config['kaggle']['timeout_minutes'] == 120
    assert args.epochs == 10 and args.patience == 3
    assert args.smoke_images == 0 and args.max_steps == 0
