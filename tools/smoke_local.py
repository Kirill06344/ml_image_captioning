"""Offline smoke of the configured architecture on synthetic RGB fixtures, never COCO results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import write_json
from src.train import main as train
from tools.prepare_data import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('runs/local-synthetic-smoke'))
    args = p.parse_args()
    if args.out.exists():
        p.error('Используйте новый --out, чтобы сохранить предыдущий результат')
    from PIL import Image, ImageDraw
    raw = args.out / 'fixture/raw'
    colors = [('red', (220, 40, 30)), ('blue', (30, 80, 220)), ('green', (40, 180, 50)), ('yellow', (220, 200, 20))]
    for source, keys in [('train2014', range(1, 17)), ('val2014', range(17, 35))]:
        (raw / source).mkdir(parents=True)
        payload = {'images': [], 'annotations': []}
        for key in keys:
            color, rgb = colors[key % len(colors)]
            image = Image.new('RGB', (256, 256), color=(245, 245, 245))
            draw = ImageDraw.Draw(image)
            left, top = 30 + key % 20, 40 + key % 15
            draw.rectangle((left, top, left + 120, top + 100), fill=rgb)
            filename = f'synthetic-{key}.jpg'
            image.save(raw / source / filename)
            payload['images'].append({'id': key, 'file_name': filename})
            for text in [f'a {color} rectangle on a white background', f'a picture of a {color} shape',
                         f'there is a {color} rectangle', f'a {color} block in the image', f'a simple {color} shape']:
                payload['annotations'].append({'image_id': key, 'caption': text})
        write_json(raw / (source + '.json'), payload)
    data = args.out / 'fixture/data'
    manifest = prepare(raw / 'train2014.json', raw / 'val2014.json', raw, data, 16, 16, 2)
    manifest.pop('fingerprint')
    manifest['dataset'] = 'Synthetic colored rectangles — NOT MS-COCO'
    manifest['synthetic_fixture'] = True
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(data / 'manifest.json', manifest)
    train(['--data', str(data), '--out', str(args.out), '--run', args.out.name,
           '--model', 'resnet18-transformer', '--dim', '256', '--layers', '2', '--max-target', '32',
           '--batch', '4', '--epochs', '4', '--max-steps', '10', '--smoke-images', '16', '--random-encoder'])
    print('Synthetic smoke complete. This verifies mechanics; these are not COCO benchmark results.')


if __name__ == '__main__':
    main()
