"""Describe one local RGB image with a saved coursework checkpoint, without downloading weights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/caption-control/best.pt'))
    args = p.parse_args()
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights
    from src.train import restore
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, saved = restore(args.checkpoint, device)
    with Image.open(args.image) as image:
        pixels = ResNet18_Weights.IMAGENET1K_V1.transforms()(image.convert('RGB'))[None].to(device)
    with torch.no_grad():
        features = model.encode_images(pixels)
        ids = model.generate(features, beams=saved['beams'])[0]
    print(vocab.decode(ids))


if __name__ == '__main__':
    main()
