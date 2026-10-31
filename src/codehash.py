"""sha256 пакета обучения — общий для трейнера и сборщика ядра.

Код едет на Kaggle приватным датасетом, а датасет можно забыть обновить, и тогда обучение молча пойдёт по старой версии.
Поэтому `tools/kaggle_train.py` считает хэш локально и передаёт его ядру ключом `--expect-code-hash`, а трейнер сверяет
его на первой секунде. Модуль без зависимостей: его импортирует и сборщик ядра, которому torch не нужен.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def package_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob('*.py') if '__pycache__' not in p.parts)


def package_hash(folder: Path) -> str:
    """Хэш по всем файлам пакета: путь относительно корня плюс содержимое."""
    digest = hashlib.sha256()
    for path in package_files(folder):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
