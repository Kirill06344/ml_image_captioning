"""Neural checks require the project's full requirements; no weights are downloaded."""
import json

import pytest
import torch
from torch.nn import functional as F

from src.model import Captioner
from src.train import main, restore
from tools.prepare_data import prepare


def test_causal_mask_padding_and_frozen_encoder():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = Captioner(10, dim=16, layers=1, max_target=8, pretrained=False, dropout=0.)
    model.train()
    features = torch.randn(2, 49, 512)
    ids = torch.tensor([[1, 4, 5, 2, 0], [1, 4, 6, 7, 2]])
    logits = model(features, ids[:, :-1])
    changed = ids[:, :-1].clone()
    changed[:, -1] = 8
    torch.testing.assert_close(logits[:, :-1], model(features, changed)[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), ignore_index=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.training
    # Additional padded targets have no contribution to cross entropy.
    extra_logits = torch.cat([logits, torch.randn(2, 1, 10)], dim=1)
    extra_targets = torch.cat([ids[:, 1:], torch.zeros(2, 1, dtype=torch.long)], dim=1)
    torch.testing.assert_close(loss, F.cross_entropy(extra_logits.flatten(0, 1), extra_targets.flatten(), ignore_index=0))


def test_offline_synthetic_training_reload_and_coco_scoring(tmp_path):
    from PIL import Image
    for split, keys in [('train2014', [1, 2]), ('val2014', [3, 4, 5, 6])]:
        (tmp_path / split).mkdir()
        raw = {'images': [], 'annotations': []}
        for key in keys:
            name = f'{key}.jpg'
            Image.new('RGB', (256, 256), color=(key * 30, 50, 80)).save(tmp_path / split / name)
            raw['images'].append({'id': key, 'file_name': name})
            for text in (['a red cat sits on a mat', 'a cat is sitting on a mat'] if key % 2 else
                         ['a blue dog runs on grass', 'a dog is running outside']):
                raw['annotations'].append({'image_id': key, 'caption': text})
        (tmp_path / f'{split}.json').write_text(json.dumps(raw))
    data, out = tmp_path / 'data', tmp_path / 'smoke'
    prepare(tmp_path / 'train2014.json', tmp_path / 'val2014.json', tmp_path, data, 2, 2, 2)
    main(['--data', str(data), '--out', str(out), '--model', 'resnet18-transformer', '--dim', '16',
          '--layers', '1', '--max-target', '10', '--batch', '2', '--max-steps', '2', '--epochs', '1',
          '--min-count', '1', '--smoke-images', '2', '--random-encoder', '--threads', '2'])
    run = json.loads((out / 'run.json').read_text())
    assert run['completed'] and run['steps'] == 2 and run['checkpoint_reloaded']
    assert not (out / 'scores_val.json').exists()
    assert not (out / 'scores_test.json').exists()
    metrics = json.loads((out / 'smoke_scores.json').read_text())['metrics']
    assert {'CIDEr', 'BLEU_4', 'METEOR', 'ROUGE_L'} <= metrics.keys()
    model, vocab, saved = restore(out / 'best.pt', torch.device('cpu'))
    assert saved['smoke'] and len(vocab.words) > 4
    assert len((out / 'predictions/val.jsonl').read_text().splitlines()) == 2
