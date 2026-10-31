"""Dependency-light integrity tests; runnable with /usr/bin/python3 -m unittest."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.data import Corpus, Vocabulary, write_json, write_predictions
from src.evaluation import require_test_selection, validate_predictions
from tools.prepare_data import prepare
from tools.baselines import constant_caption
from tools.collect_results import collect
from src.config import load_config, train_argv
from src.train import build_parser


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for split, keys in [('train2014', [1, 2, 3]), ('val2014', [4, 5, 6, 7])]:
            (self.root / split).mkdir()
            images, annotations = [], []
            for key in keys:
                name = f'{key}.jpg'
                (self.root / split / name).write_bytes(b'fixture integrity bytes')
                images.append({'id': key, 'file_name': name})
                for text in ['a cat sits .', 'a small cat .']:
                    annotations.append({'image_id': key, 'caption': text})
            write_json(self.root / f'{split}.json', {'images': images, 'annotations': annotations})
        self.data = self.root / 'prepared'
        prepare(self.root / 'train2014.json', self.root / 'val2014.json', self.root, self.data, 3, 2, 2)
        self.corpus = Corpus(self.data)

    def test_disjoint_and_multi_reference(self):
        ids = [r['image_id'] for s in ['train', 'val', 'test'] for r in self.corpus.rows(s)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(self.corpus.rows('val')[0]['references']), 2)

    def test_rows_tampering_is_rejected(self):
        with (self.data / 'val.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'изменился'):
            self.corpus.rows('val')

    def test_manifest_tampering_is_rejected(self):
        manifest = self.corpus.manifest
        manifest['total_images'] = 88
        write_json(self.data / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            Corpus(self.data)

    def test_vocabulary_uses_train_only_and_masks_padding(self):
        vocab = Vocabulary.build(self.corpus.rows('train'))
        encoded = vocab.encode('unseen word', 8)
        self.assertEqual(encoded, [1, 3, 3, 2, 0, 0, 0, 0])
        self.assertEqual(vocab.encode('cat ' * 100, 4), [1, vocab.index['cat'], vocab.index['cat'], 2])
        self.assertEqual(vocab.decode([1, vocab.index['cat'], vocab.index['.'], 2]), 'cat.')

    def predictions(self):
        rows = self.corpus.rows('val')
        return rows, [dict(row, prediction='a cat sits .') for row in rows]

    def test_duplicate_missing_extra_predictions_rejected(self):
        rows, predictions = self.predictions()
        for invalid in [predictions[:1], predictions + predictions[:1],
                        [dict(predictions[0], image_id=999), predictions[1]]]:
            with self.assertRaisesRegex(ValueError, 'покрытие'):
                validate_predictions(rows, invalid)

    def test_references_cannot_be_replaced(self):
        rows, predictions = self.predictions()
        predictions[0] = dict(predictions[0], references=['a fabricated reference'])
        with self.assertRaisesRegex(ValueError, 'эталоны'):
            validate_predictions(rows, predictions)

    def test_prediction_order_is_normalized(self):
        rows, predictions = self.predictions()
        self.assertEqual(validate_predictions(rows, predictions[::-1]), predictions)

    def test_constant_ignores_held_references(self):
        train = self.corpus.rows('train')
        self.assertEqual(constant_caption(train), 'a cat sits .')
        self.assertNotIn('unseen', constant_caption(train))

    def test_predictions_written_with_all_references(self):
        rows = self.corpus.rows('val')
        path = self.root / 'predictions.jsonl'
        write_predictions(path, rows, ['one', 'two'])
        decoded = [json.loads(s) for s in path.read_text().splitlines()]
        self.assertEqual(validate_predictions(rows, decoded), decoded)
        with self.assertRaises(ValueError):
            write_predictions(path, rows, ['one'])

    def test_test_gate(self):
        fingerprint = self.corpus.manifest['fingerprint']
        selection = self.root / 'selection.json'
        write_json(selection, {'split': 'val', 'final': 'winner', 'roles': {'winner': 'candidate', 'b': 'baseline'},
                               'dataset_fingerprint': fingerprint})
        require_test_selection(selection, 'winner', 'candidate', fingerprint)
        require_test_selection(selection, 'b', 'baseline', fingerprint)
        for run, role, fp in [('loser', 'candidate', fingerprint), ('winner', 'baseline', fingerprint),
                              ('winner', 'candidate', 'another-corpus')]:
            with self.assertRaises(ValueError):
                require_test_selection(selection, run, role, fp)
        with self.assertRaises(ValueError):
            require_test_selection(None, 'winner', 'candidate', fingerprint)

    def test_config_round_trip(self):
        for path in Path('configs/experiments').glob('*.yaml'):
            config = load_config(path)
            args = build_parser().parse_args(['--data', 'in', '--out', 'out',
                                               *train_argv(build_parser(), config['train'])])
            self.assertEqual(args.model, 'resnet18-transformer')
        with self.assertRaises(ValueError):
            train_argv(build_parser(), {'run': 'hidden-run'})

    def test_zip_image_payload_integrity(self):
        # Stop at Pillow import on systems without imaging dependencies; byte reading is tested by hash.
        row = self.corpus.rows('val')[0]
        source = self.data / row['image']
        with zipfile.ZipFile(self.data / 'images.zip', 'w') as archive:
            archive.write(source, Path(row['image']).relative_to('images').as_posix())
        with zipfile.ZipFile(self.data / 'images.zip') as archive:
            content = archive.read(Path(row['image']).relative_to('images').as_posix())
        self.assertEqual(hashlib.sha256(content).hexdigest(), row['image_sha256'])

    def test_reference_cannot_win_selection(self):
        for run, role, score in [('candidate', 'candidate', .5), ('reference', 'reference', 9.), ('constant', 'baseline', .1)]:
            report = {'run': run, 'role': role, 'split': 'val', 'n_images': 2,
                      'dataset_fingerprint': 'fixture', 'image_ids_sha256': 'fixture',
                      'scorer': {'name': 'fixture'}, 'metric_units': {'CIDEr': 'raw'}, 'metrics': {'CIDEr': score}}
            write_json(self.root / 'runs' / run / 'scores_val.json', report)
        _, selection = collect(self.root / 'runs', 'val', 'CIDEr', self.root / 'results')
        self.assertEqual(selection['final'], 'candidate')


if __name__ == '__main__':
    unittest.main()
