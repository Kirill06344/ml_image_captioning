"""Verified image-level COCO manifests and a train-only word vocabulary."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def find_data(path):
    path = Path(path)
    candidates = [path / 'manifest.json'] if (path / 'manifest.json').is_file() else sorted(path.rglob('manifest.json'))
    candidates = [p for p in candidates if all((p.parent / f'{s}.jsonl').is_file() for s in ('train', 'val', 'test'))]
    if len(candidates) != 1:
        raise ValueError(f'Ожидается один COCO manifest, найдено {len(candidates)} в {path}')
    return candidates[0].parent


class Corpus:
    def __init__(self, root):
        self.root = find_data(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        unsigned = {k: v for k, v in self.manifest.items() if k != 'fingerprint'}
        expected = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if self.manifest['fingerprint'] != expected:
            raise ValueError('Неверный fingerprint manifest')
        ids = [i for split in self.manifest['splits'].values() for i in split['image_ids']]
        if len(ids) != len(set(ids)) or len(ids) != self.manifest['total_images']:
            raise ValueError('Пересечение сплитов или неверное число изображений')

    def rows(self, split):
        path = self.root / f'{split}.jsonl'
        spec = self.manifest['splits'][split]
        if sha256(path) != spec['rows_sha256']:
            raise ValueError(f'{split}: изменился JSONL')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        ids = [r['image_id'] for r in rows]
        if ids != spec['image_ids'] or len(rows) != spec['n_images']:
            raise ValueError(f'{split}: неполные или повторные image_id')
        for row in rows:
            relative = Path(row['image'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Небезопасный путь изображения')
            if not isinstance(row['references'], list) or not row['references'] or any(
                    not isinstance(t, str) or not t.strip() for t in row['references']):
                raise ValueError('Отсутствуют эталоны')
        return rows

    def image(self, row):
        """Read-only support for Kaggle's images.zip, without extraction into /kaggle/input."""
        from PIL import Image
        import io
        relative = Path(row['image'])
        path = self.root / relative
        if path.is_file():
            payload = path.read_bytes()
        else:
            with zipfile.ZipFile(self.root / 'images.zip') as archive:
                names = [relative.as_posix(), Path(*relative.parts[1:]).as_posix()]
                name = next((n for n in names if n in archive.namelist()), None)
                if name is None:
                    raise FileNotFoundError(relative)
                payload = archive.read(name)
        if hashlib.sha256(payload).hexdigest() != row['image_sha256']:
            raise ValueError(f'Изменилось изображение {row["image_id"]}')
        with Image.open(io.BytesIO(payload)) as im:
            return im.convert('RGB')


def tokens(text):
    # Model vocabulary tokenizer; scoring uses Stanford PTB separately.
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\w\s]", text.lower())


class Vocabulary:
    specials = ['<pad>', '<bos>', '<eos>', '<unk>']

    def __init__(self, words):
        self.words = list(words)
        if self.words[:4] != self.specials or len(set(self.words)) != len(self.words):
            raise ValueError('Некорректный словарь')
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def build(cls, rows, min_count=2):
        counts = Counter(t for r in rows for ref in r['references'] for t in tokens(ref))
        return cls(cls.specials + sorted(w for w, n in counts.items() if n >= min_count))

    def encode(self, text, length):
        seq = [1] + [self.index.get(w, 3) for w in tokens(text)][:length - 2] + [2]
        return seq + [0] * (length - len(seq))

    def decode(self, ids):
        words = []
        for i in ids:
            if i == 2:
                break
            if i >= 4:
                words.append(self.words[i])
        return re.sub(r'\s+([.,!?;:])', r'\1', ' '.join(words))


def write_predictions(path, rows, captions):
    if len(rows) != len(captions):
        raise ValueError('Неполные прогнозы')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as stream:
        for row, caption in zip(rows, captions):
            stream.write(json.dumps({k: row[k] for k in ('image_id', 'image', 'references')} |
                                    {'prediction': caption}, ensure_ascii=False) + '\n')
