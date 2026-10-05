"""Dataset loading and stable scenario identity shared by every pipeline."""
import json
from pathlib import Path

from datasets import load_dataset


def normalize_id(value) -> str:
    if type(value) is int:
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError('Every scenario requires a nonempty string or integer id')


def index_by_id(records, label='records') -> dict[str, dict]:
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f'{label} must contain objects')
        identifier = normalize_id(record.get('id'))
        if identifier in indexed:
            raise ValueError(f'Duplicate id in {label}: {identifier}')
        indexed[identifier] = record
    return indexed


def load_dataset_wrapper(dataset_repo: str, split: str = 'train', cache_dir: str | None = None):
    path = Path(dataset_repo)
    if path.is_file() or path.suffix in ('.json', '.jsonl'):
        with path.open(encoding='utf-8') as handle:
            if path.suffix == '.jsonl':
                rows = [json.loads(line) for line in handle if line.strip()]
            else:
                rows = json.load(handle)
                if isinstance(rows, dict):
                    rows = rows[split]
        if not isinstance(rows, list):
            raise ValueError('Local dataset must be a JSON array or JSONL records')
        index_by_id(rows, 'dataset')
        return rows
    return load_dataset(dataset_repo, split=split, cache_dir=cache_dir)
