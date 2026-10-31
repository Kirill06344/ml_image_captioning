"""Train-only constant caption and nearest-image retrieval baselines."""
from pathlib import Path
import argparse
from collections import Counter
import sys
import os
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import Corpus, tokens, write_predictions, write_json
from src.evaluation import require_test_selection, score_predictions


def constant_caption(rows):
    counts = Counter(' '.join(tokens(text)) for r in rows for text in r['references'])
    return min(counts, key=lambda text: (-counts[text], text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=['constant', 'retrieval'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--run', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--selection', type=Path)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()
    started = time.monotonic()
    corpus = Corpus(args.data)
    if args.split == 'test':
        require_test_selection(args.selection, args.run, 'baseline', corpus.manifest['fingerprint'])
    train, held = corpus.rows('train'), corpus.rows(args.split)
    provenance = {'kind': args.kind, 'source_split': 'train', 'split': args.split,
                  'dataset_fingerprint': corpus.manifest['fingerprint']}
    if args.kind == 'constant':
        text = constant_caption(train)
        captions = [text] * len(held)
        provenance['constant'] = text
    else:
        import torch
        from torch.nn.functional import normalize
        from src.train import restore, extract_features
        torch.set_num_threads(4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.checkpoint:
            model, _, saved = restore(args.checkpoint, device)
            if saved['smoke'] or saved['dataset_fingerprint'] != corpus.manifest['fingerprint']:
                raise ValueError('Retrieval требует полный checkpoint того же датасета')
            provenance['encoder_source'] = str(args.checkpoint)
        else:
            # Only the frozen encoder is used. The decoder is never called by retrieval.
            from src.model import Captioner
            os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
            model = Captioner(5, dim=16, layers=1, pretrained=True).to(device).eval()
            provenance['encoder_source'] = 'torchvision ResNet18_Weights.IMAGENET1K_V1'
        print('Извлечение train ImageNet-признаков:', len(train), flush=True)
        features = normalize(extract_features(corpus, train, model, device, args.batch).mean(1), dim=1)
        print('Извлечение признаков', args.split, len(held), flush=True)
        query = normalize(extract_features(corpus, held, model, device, args.batch).mean(1), dim=1)
        neighbors = (query @ features.T).argmax(1).tolist()
        # The annotation order is fixed by the original JSON; no held references participate.
        captions = [train[i]['references'][0] for i in neighbors]
        provenance['neighbors'] = {str(r['image_id']): train[i]['image_id'] for r, i in zip(held, neighbors)}
    path = args.out / 'predictions' / f'{args.split}.jsonl'
    write_predictions(path, held, captions)
    score_predictions(corpus, args.split, path, args.out, args.run, role='baseline', selection=args.selection)
    provenance['elapsed_seconds'] = time.monotonic() - started
    write_json(args.out / f'baseline_{args.split}.json', provenance)


if __name__ == '__main__':
    main()
