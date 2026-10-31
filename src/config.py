"""YAML-конфиги экспериментов: один файл — один прогон.

    # configs/experiments/smoke.yaml
    extends: ../base.yaml          # путь относительно этого файла, цепочки можно
    note: дымовой тест captioning
    train:
      model: resnet18-transformer

Секции сливаются рекурсивно, дочерний файл перекрывает родителя. Ключи `train` — это ключи `src.train` без `--` и с `_`
вместо `-`; неизвестный ключ — ошибка сразу, а не молча проигнорированный флаг на Kaggle. Слаг прогона —
`caption-<имя файла>` либо явный `slug:`. `tools/kaggle_train.py` разворачивает конфиг в явные ключи: командная строка ядра
остаётся единственным источником правды о прогоне, а разрешённый конфиг ложится в `runs/<slug>/config.yaml`.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

SECTIONS = {'extends', 'slug', 'note', 'kaggle', 'train'}
KAGGLE_KEYS = {'data_slug', 'timeout_minutes', 'internet'}
# Ключи трейнера, которые ставит сам запуск, а не эксперимент.
LAUNCH_ONLY = {'data', 'out', 'require_gpu', 'expect_code_hash', 'run'}


def _merge(base: dict, child: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path, _seen: tuple = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f'цикл extends: {" -> ".join(map(str, (*_seen, path)))}')
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: конфиг должен быть словарём')
    unknown = set(raw) - SECTIONS
    if unknown:
        raise ValueError(f'{path}: неизвестные секции {sorted(unknown)}, есть {sorted(SECTIONS)}')
    unknown = set(raw.get('kaggle') or {}) - KAGGLE_KEYS
    if unknown:
        raise ValueError(f'{path}: неизвестные ключи kaggle {sorted(unknown)}')
    parent = {}
    if raw.get('extends'):
        parent = load_config(path.parent / raw['extends'], (*_seen, path))
    own = {k: v for k, v in raw.items() if k not in ('extends', 'slug')}
    config = _merge(parent, own)
    config.setdefault('kaggle', {})
    config.setdefault('train', {})
    # Слаг не наследуется: у каждого файла свой прогон.
    config['slug'] = raw.get('slug') or f'caption-{path.stem}'
    config['path'] = str(path)
    return config


def _actions(parser: argparse.ArgumentParser) -> dict:
    return {a.dest: a for a in parser._actions if a.option_strings and a.dest != 'help'}


def train_argv(parser: argparse.ArgumentParser, train: dict) -> list[str]:
    """Развернуть секцию `train` в явные ключи командной строки трейнера."""
    actions = _actions(parser)
    unknown = set(train) - set(actions)
    if unknown:
        raise ValueError(f'неизвестные ключи train: {sorted(unknown)}')
    forbidden = set(train) & LAUNCH_ONLY
    if forbidden:
        raise ValueError(f'ключи {sorted(forbidden)} ставит запуск, не конфиг')
    argv = []
    for key in sorted(train):
        value, action = train[key], actions[key]
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ValueError(f'{key}: ожидается true/false, а не {value!r}')
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            value = ','.join(str(v) for v in value)
        argv += [flag, str(value)]
    return argv
