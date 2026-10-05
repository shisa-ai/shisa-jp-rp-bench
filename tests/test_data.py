import json
import pytest
from japanese_rp_bench import data


def test_normalized_ids_preserve_zero_and_match_numeric_string():
    assert data.normalize_id(0) == '0'
    assert data.normalize_id('0') == '0'
    assert data.index_by_id([{'id': 0, 'text': 'zero'}, {'id': 'a'}])['0']['text'] == 'zero'


@pytest.mark.parametrize('value', [None, '', '  ', True, False, 1.5, {}, []])
def test_invalid_ids_rejected(value):
    with pytest.raises(ValueError):
        data.normalize_id(value)


@pytest.mark.parametrize('rows', [[{'id': 0}, {'id':'0'}], [{}], [None]])
def test_index_rejects_duplicate_missing_or_nonobject_records(rows):
    with pytest.raises(ValueError):
        data.index_by_id(rows)


@pytest.mark.parametrize('suffix', ['json', 'jsonl'])
def test_local_dataset_needs_no_download(tmp_path, monkeypatch, suffix):
    rows = [{'id': 0, 'text': '日本語'}, {'id': 'a'}]
    path = tmp_path / ('dataset.' + suffix)
    path.write_text(json.dumps(rows) if suffix == 'json' else '\n'.join(json.dumps(r) for r in rows) + '\n\n')
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: pytest.fail('Local file attempted download'))
    assert data.load_dataset_wrapper(str(path), 'train', None) == rows


def test_local_split_mapping_selects_requested_split(tmp_path):
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps({'train': [{'id': 0}], 'test': [{'id': 1}]}))
    assert data.load_dataset_wrapper(str(path), 'test') == [{'id': 1}]


@pytest.mark.parametrize('value', [None, 3, 'bad'])
def test_local_nonarray_dataset_rejected(tmp_path, value):
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='JSON array'):
        data.load_dataset_wrapper(str(path))


def test_remote_dataset_passes_split_and_cache(monkeypatch):
    calls = []
    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        return [{'id': 0}]
    monkeypatch.setattr(data, 'load_dataset', load)
    assert data.load_dataset_wrapper('org/scenarios', 'test', '/cache') == [{'id': 0}]
    assert calls == [('org/scenarios', {'split': 'test', 'cache_dir': '/cache'})]
