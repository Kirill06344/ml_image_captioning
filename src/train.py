"""Train a frozen ResNet18 + Transformer captioner, select checkpoints on val CIDEr."""
from __future__ import annotations

import argparse
from pathlib import Path

from .codehash import package_hash

IMPLEMENTED = True


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['resnet18-transformer'], required=True)
    p.add_argument('--revision', default='IMAGENET1K_V1')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-target', type=int, default=32)
    p.add_argument('--beams', type=int, default=1)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--min-count', type=int, default=2)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--selection', type=Path)
    p.add_argument('--run', default=None)
    p.add_argument('--smoke-images', type=int, default=0, help='train/val only; marks output as smoke')
    p.add_argument('--random-encoder', action='store_true', help='synthetic offline smoke only')
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--expect-code-hash', default=None)
    return p


def extract_features(corpus, rows, model, device, batch):
    import torch
    from torchvision.models import ResNet18_Weights
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    chunks = []
    model.eval()
    for start in range(0, len(rows), batch):
        images = torch.stack([transform(corpus.image(row)) for row in rows[start:start + batch]])
        chunks.append(model.encode_images(images.to(device)).cpu())
    return torch.cat(chunks)


def restore(path, device):
    import torch
    from .model import Captioner
    from .data import Vocabulary
    # Only load checkpoints produced by this project. weights_only excludes arbitrary pickle objects.
    saved = torch.load(path, map_location='cpu', weights_only=True)
    vocab = Vocabulary(saved['vocabulary'])
    model = Captioner(len(vocab.words), **saved['architecture'], pretrained=False)
    model.load_state_dict(saved['state_dict'])
    model.to(device).eval()
    return model, vocab, saved


def generate(model, vocab, features, rows, device, beams, batch):
    captions = []
    model.eval()
    for start in range(0, len(rows), batch):
        ids = model.generate(features[start:start + batch].to(device), beams)
        captions.extend(vocab.decode(seq) for seq in ids)
    return captions


def main(argv=None):
    args = build_parser().parse_args(argv)
    digest = package_hash(Path(__file__).parent)
    if args.expect_code_hash and args.expect_code_hash != digest:
        raise SystemExit('Хэш пакета не совпадает с ожидаемым')
    if args.batch < 1 or args.epochs < 1 or args.max_target < 3 or args.dim % 4 or args.dim < 4:
        raise SystemExit('Некорректные размеры batch/epochs/max_target/dim')
    if args.beams < 1 or args.lr <= 0 or args.layers < 1 or args.max_steps < 0 or args.patience < 1:
        raise SystemExit('Некорректный рецепт обучения')
    if args.random_encoder and not args.smoke_images:
        raise SystemExit('--random-encoder разрешён только для smoke')
    if args.split == 'test' and (not args.eval_only or args.smoke_images):
        raise SystemExit('Test доступен только --eval-only после выбора по val')
    if args.smoke_images and args.smoke_images < 2:
        raise SystemExit('Smoke требует хотя бы два изображения в split')

    import importlib.metadata
    import os
    import json
    import random
    import time
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from .data import Corpus, Vocabulary, write_json, write_predictions, sha256
    from .evaluation import coco_metrics, require_test_selection, score_predictions, SCORER
    from .model import Captioner

    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit('GPU обязательна, CUDA недоступна')
    # Fail before allocating a model or downloading weights if the scorer cannot run.
    import shutil
    if shutil.which('java') is None or importlib.metadata.version('pycocoevalcap') != '1.2':
        raise SystemExit('Нужны Java и pycocoevalcap==1.2')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    corpus = Corpus(args.data)
    run = args.run or args.out.name
    if args.revision != 'IMAGENET1K_V1':
        raise SystemExit('Поддерживается только фиксированная revision IMAGENET1K_V1')
    if args.split == 'test':
        require_test_selection(args.selection, run, 'candidate', corpus.manifest['fingerprint'])
    if not args.eval_only and any((args.out / name).exists() for name in ('best.pt', 'history.json', 'run.json', 'predictions')):
        raise SystemExit('Каталог прогона не пуст; используйте новый slug')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke_images and (args.out / 'scores_val.json').exists():
        raise SystemExit('Smoke нельзя записывать поверх полного опыта')
    os.environ.setdefault('TORCH_HOME', str(Path.cwd() / '.cache' / 'torch'))
    started = time.monotonic()
    provenance = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'code_hash': digest, 'dataset_fingerprint': corpus.manifest['fingerprint'],
                  'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  'versions': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'pycocoevalcap')},
                  'encoder_weights': 'random, smoke only' if args.random_encoder else
                      'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                  'scorer': SCORER, 'smoke': bool(args.smoke_images)}
    provenance_path = args.out / ('evaluation_' + args.split + '.json' if args.eval_only else 'run.json')
    write_json(provenance_path, provenance)

    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit('--eval-only требует --checkpoint')
        model, vocab, saved = restore(args.checkpoint, device)
        if saved['dataset_fingerprint'] != corpus.manifest['fingerprint'] or saved['smoke'] or saved['run'] != run:
            raise SystemExit('Checkpoint другого датасета или smoke запрещён для итоговой оценки')
        if args.beams != saved['beams']:
            raise SystemExit('Генерация должна совпадать с рецептом выбора checkpoint')
        rows = corpus.rows(args.split)
        features = extract_features(corpus, rows, model, device, args.batch)
        path = args.out / 'predictions' / f'{args.split}.jsonl'
        write_predictions(path, rows, generate(model, vocab, features, rows, device, args.beams, args.batch))
        score_predictions(corpus, args.split, path, args.out, run, selection=args.selection)
    else:
        train_rows, val_rows = corpus.rows('train'), corpus.rows('val')
        if args.smoke_images:
            train_rows, val_rows = train_rows[:args.smoke_images], val_rows[:args.smoke_images]
        vocab = Vocabulary.build(train_rows, args.min_count)
        if len(vocab.words) < 5:
            raise SystemExit('Словарь пуст; проверьте подписи или --min-count')
        architecture = {'dim': args.dim, 'layers': args.layers, 'max_target': args.max_target}
        model = Captioner(len(vocab.words), **architecture, pretrained=not args.random_encoder).to(device)
        train_features = extract_features(corpus, train_rows, model, device, args.batch)
        val_features = extract_features(corpus, val_rows, model, device, args.batch)
        pairs = [(i, vocab.encode(ref, args.max_target)) for i, row in enumerate(train_rows) for ref in row['references']]
        dataset = TensorDataset(torch.tensor([i for i, _ in pairs]), torch.tensor([t for _, t in pairs]))
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=generator)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
        best, best_epoch, steps, stale = -float('inf'), 0, 0, 0
        history = []
        best_path = args.out / 'best.pt'
        for epoch in range(args.epochs):
            model.train()
            loss_sum, token_count, updates = 0., 0, 0
            for indices, targets in loader:
                features = train_features[indices].to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features, targets[:, :-1])
                loss = F.cross_entropy(logits.flatten(0, 1), targets[:, 1:].flatten(), ignore_index=0)
                if not torch.isfinite(loss):
                    raise RuntimeError('Loss не finite; опыт остановлен')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Градиенты не finite; опыт остановлен')
                optimizer.step()
                steps += 1
                updates += 1
                count = int(targets[:, 1:].ne(0).sum())
                loss_sum += float(loss.detach()) * count
                token_count += count
                if args.max_steps and steps >= args.max_steps:
                    break
            captions = generate(model, vocab, val_features, val_rows, device, args.beams, args.batch)
            predictions = [dict(row, prediction=text) for row, text in zip(val_rows, captions)]
            metric, _ = coco_metrics(predictions, cider_only=True)
            record = {'epoch': epoch + 1, 'steps': steps, 'updates': updates, 'loss': loss_sum / token_count,
                      'val_CIDEr': metric['CIDEr'], 'elapsed_seconds': time.monotonic() - started}
            history.append(record)
            print(json.dumps(record), flush=True)
            write_json(args.out / 'history.json', history)
            if metric['CIDEr'] > best:
                best, best_epoch, stale = metric['CIDEr'], epoch + 1, 0
                state = {'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                         'vocabulary': vocab.words, 'architecture': architecture,
                         'dataset_fingerprint': corpus.manifest['fingerprint'], 'smoke': bool(args.smoke_images),
                         'epoch': best_epoch, 'val_CIDEr': best, 'code_hash': digest,
                         'run': run, 'beams': args.beams}
                torch.save(state, best_path)
            else:
                stale += 1
            if stale >= args.patience or (args.max_steps and steps >= args.max_steps):
                break
        # Reload and compare real logits; checkpoints include the frozen encoder and vocabulary.
        restored, restored_vocab, saved = restore(best_path, device)
        if saved['epoch'] == history[-1]['epoch']:
            model.eval()
            probe = torch.tensor([[1]], device=device)
            with torch.no_grad():
                torch.testing.assert_close(model(val_features[:1].to(device), probe),
                                           restored(val_features[:1].to(device), probe))
        captions = generate(restored, restored_vocab, val_features, val_rows, device, args.beams, args.batch)
        path = args.out / 'predictions' / 'val.jsonl'
        write_predictions(path, val_rows, captions)
        if args.smoke_images:
            metrics, _ = coco_metrics([dict(row, prediction=t) for row, t in zip(val_rows, captions)])
            write_json(args.out / 'smoke_scores.json', {'smoke': True, 'n_images': len(val_rows), 'metrics': metrics})
        else:
            score_predictions(corpus, 'val', path, args.out, run)
        provenance.update({'steps': steps, 'best_epoch': best_epoch, 'best_checkpoint_sha256': sha256(best_path),
                           'train_images': len(train_rows), 'train_captions': len(pairs), 'val_images': len(val_rows),
                           'vocabulary_size': len(vocab.words), 'checkpoint_reloaded': True})
    if device.type == 'cuda':
        torch.cuda.synchronize()
    provenance['elapsed_seconds'] = time.monotonic() - started
    provenance['gpu_hours'] = provenance['elapsed_seconds'] / 3600 if device.type == 'cuda' else 0
    provenance['peak_gpu_bytes'] = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    provenance['completed'] = True
    write_json(provenance_path, provenance)


if __name__ == '__main__':
    main()
