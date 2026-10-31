"""Score a complete split using COCO PTB + pycocoevalcap 1.2 (Java required)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Corpus
from src.evaluation import score_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--role', choices=['candidate', 'baseline', 'reference'], default='candidate')
    p.add_argument('--selection', type=Path)
    args = p.parse_args()
    print(score_predictions(Corpus(args.data), args.split, args.predictions, args.out,
                            args.run, args.role, args.selection)['metrics'])


if __name__ == '__main__':
    main()
