"""Character identity, JSON extraction, and score-validation contracts."""
import json
import logging

import pytest

from japanese_rp_bench import data, prompts, utils


@pytest.fixture
def scenario():
    return {
        "id": "scene-1", "tag": "SFW", "genre": "冒険", "world_setting": "森",
        "scene_setting": "朝", "user_setting": "旅人", "assistant_setting": "案内人",
        "dialogue_tone": "親切", "response_format": "台詞のみ", "first_user_input": "こんにちは",
    }


def test_prompt_characters_are_swapped_for_user_model(scenario):
    assistant, user, initial = prompts.construct_system_prompts(scenario)
    assert initial == "こんにちは"
    assert '### あなたがなりきる人物の設定:\n案内人' in assistant
    assert '### ユーザーがなりきる人物の設定:\n旅人' in assistant
    assert '### あなたがなりきる人物の設定:\n旅人' in user
    assert '### ユーザーがなりきる人物の設定:\n案内人' in user
    for value in ["SFW", "冒険", "森", "朝", "親切", "台詞のみ"]:
        assert value in assistant and value in user
    assert "展開役" in user
    assert scenario["user_setting"] == "旅人"


@pytest.mark.parametrize("field", ["tag", "genre", "world_setting", "scene_setting", "user_setting",
    "assistant_setting", "dialogue_tone", "response_format", "first_user_input"])
def test_missing_prompt_fields_raise(scenario, field):
    del scenario[field]
    with pytest.raises(KeyError, match=field):
        prompts.construct_system_prompts(scenario)


@pytest.mark.parametrize(("raw", "expected"), [
    ('prefix {"score": 5} suffix', {"score": 5}),
    ('```json\n{"nested": {"score": 3}}\n```', {"nested": {"score": 3}}),
    ('[1, 2, 3]', [1, 2, 3]),
    ('{"reason": "一行目\n二行目"}', {"reason": "一行目\n二行目"}),
    ('{"a": 1} and {"b": 2}', {"a": 1}),
])
def test_extract_json(raw, expected):
    assert json.loads(utils.extract_and_escape_json_string(raw)) == expected


def test_extract_missing_json_raises():
    with pytest.raises(IndexError):
        utils.extract_and_escape_json_string("not JSON")


@pytest.mark.parametrize("text", ['{"reason": "a } b", "score": 3}', '{"reason": "a { b", "score": 3}'])
def test_extract_preserves_braces_inside_json_strings(text):
    assert json.loads(utils.extract_and_escape_json_string(text)) == json.loads(text)


CATEGORIES = ["Roleplay Adherence", "Consistency", "Contextual Understanding", "Expressiveness",
              "Creativity", "Naturalness of Japanese", "Enjoyment of the Dialogue", "Appropriateness of Turn-Taking"]


def valid_evaluation(score=3):
    return {"Evaluation Reason": "一貫性がある", **dict.fromkeys(CATEGORIES, score)}


def test_valid_evaluation():
    assert utils.is_valid_evaluation(valid_evaluation())


@pytest.mark.parametrize("category", CATEGORIES)
def test_evaluation_requires_all_categories(category):
    evaluation = valid_evaluation()
    del evaluation[category]
    assert not utils.is_valid_evaluation(evaluation)


@pytest.mark.parametrize("score", [None, True, "5", -1, 0, 6, float("nan"), float("inf"), [], {}])
def test_evaluation_rejects_invalid_score_values(score):
    assert not utils.is_valid_evaluation(valid_evaluation(score))


def test_evaluation_requires_reason():
    assert not utils.is_valid_evaluation(dict.fromkeys(CATEGORIES, 3))


def test_evaluation_must_be_mapping():
    assert not utils.is_valid_evaluation(CATEGORIES)


def test_data_loader_forwards_dataset_settings(monkeypatch):
    received = []
    def load(repo, **kwargs):
        received.append((repo, kwargs))
        return [{"id": "a"}]
    monkeypatch.setattr(data, "load_dataset", load)
    assert data.load_dataset_wrapper("org/data", "test", "/tmp/cache") == [{"id": "a"}]
    assert received == [("org/data", {"split": "test", "cache_dir": "/tmp/cache"})]


def test_data_loader_propagates_download_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(data, "load_dataset", fail)
    with pytest.raises(OSError, match="offline"):
        data.load_dataset_wrapper("org/data", "test", None)


def test_logging_sets_expected_levels():
    names = ["httpx", "japanese_rp_bench"]
    before = {name: logging.getLogger(name).level for name in names}
    try:
        logger = utils.setup_logging()
        assert logger.name == "japanese_rp_bench"
        assert logger.level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
    finally:
        for name, level in before.items():
            logging.getLogger(name).setLevel(level)


@pytest.mark.parametrize('extra', [float('nan'), float('inf'), {'nested': [float('-inf')]}])
def test_nonfinite_extra_fields_cannot_break_artifact_serialization(extra):
    evaluation = valid_evaluation()
    evaluation['extra'] = extra
    assert not utils.is_valid_evaluation(evaluation)
    with pytest.raises(ValueError):
        utils.parse_evaluation(evaluation)
