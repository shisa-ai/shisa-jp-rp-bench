"""JSON parsing and the shared, strict eight-category evaluation contract."""
import json
import logging

EVALUATION_CATEGORIES = (
    'Roleplay Adherence', 'Consistency', 'Contextual Understanding',
    'Expressiveness', 'Creativity', 'Naturalness of Japanese',
    'Enjoyment of the Dialogue', 'Appropriateness of Turn-Taking',
)


def setup_logging():
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logger = logging.getLogger('japanese_rp_bench')
    logger.setLevel(logging.WARNING)
    return logger


def extract_and_escape_json_string(input_string):
    """Extract one JSON container while respecting quoted braces and escapes.

    strict=False permits literal newlines in generated strings; serialization
    escapes them again. Malformed containers are never repaired by deleting data.
    """
    if not isinstance(input_string, str):
        raise ValueError('Evaluation text must be a string')
    decoder = json.JSONDecoder(strict=False)
    found = False
    for offset, char in enumerate(input_string):
        if char not in '{[':
            continue
        found = True
        try:
            value, _ = decoder.raw_decode(input_string[offset:])
        except json.JSONDecodeError:
            continue
        return json.dumps(value, ensure_ascii=False)
    if not found:
        raise IndexError('No JSON container found')
    raise ValueError('No valid JSON container found')


def is_valid_evaluation(evaluation):
    try:
        json.dumps(evaluation, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(evaluation, dict)
        and isinstance(evaluation.get('Evaluation Reason'), str)
        and bool(evaluation['Evaluation Reason'].strip())
        and all(type(evaluation.get(key)) is int and 1 <= evaluation[key] <= 5
                for key in EVALUATION_CATEGORIES)
    )


def parse_evaluation(text_or_dict) -> dict:
    try:
        result = text_or_dict if isinstance(text_or_dict, dict) else json.loads(extract_and_escape_json_string(text_or_dict))
    except (IndexError, ValueError, TypeError):
        raise ValueError('Judge evaluation must contain valid JSON') from None
    if not is_valid_evaluation(result):
        raise ValueError('Judge evaluation requires a reason and all eight integer scores from 1 to 5')
    return result
