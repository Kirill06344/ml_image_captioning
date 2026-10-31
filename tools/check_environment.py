"""Read-only preflight; prints package versions and readiness, never credentials."""
import importlib.metadata
from pathlib import Path
import shutil
import sys


def main():
    failed = []
    print('Python:', sys.executable)
    for package in ['torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML', 'pycocoevalcap',
                    'pytest', 'python-docx', 'markdown-it-py', 'matplotlib', 'kaggle']:
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = 'MISSING'
            failed.append(package)
        print(f'{package}: {value}')
        if package == 'pycocoevalcap' and value not in ('1.2', 'MISSING'):
            failed.append('pycocoevalcap must be 1.2')
    print('Java:', shutil.which('java') or 'MISSING')
    if not shutil.which('java'):
        failed.append('Java')
    print('LibreOffice:', shutil.which('libreoffice') or 'MISSING on host (Docker rendering: docs/runbook.md)')
    print('Prepared COCO:', (Path('data/processed') / 'manifest.json').is_file())
    if failed:
        print('Dependencies not ready:', ', '.join(failed))
        return 1
    try:
        import torch
        import torchvision
        print('CUDA:', torch.cuda.is_available())
        print('torchvision import: OK')
    except Exception as exc:
        print('Neural import FAILED:', type(exc).__name__, str(exc))
        return 1
    print('Dependencies ready. Run pytest and smoke before experiments.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
