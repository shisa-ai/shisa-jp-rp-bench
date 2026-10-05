"""Offline export and reaggregation of namespaced absolute score artifacts."""
import json
from pathlib import Path

import click

from .artifacts import atomic_write, safe_name, score_paths, write_json
from .data import index_by_id
from .utils import EVALUATION_CATEGORIES, parse_evaluation


def export_scores(source, destination):
    count = 0
    for path in sorted(source.glob("*/*/scores.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model, judge = data["model_name"], data["judge_model_name"]
        if path != score_paths(model, judge, source)["scores"]:
            raise ValueError(f"Score metadata does not match its namespace: {path}")
        if data.get("status") != "complete" or data.get("overall_average") is None:
            continue
        output = destination / safe_name(model) / safe_name(judge) / "score-shisa-jp-rp-bench.jsonl"
        atomic_write(output, json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        count += 1
    return count


def reaggregate_scores(source):
    count = 0
    failed = 0
    for path in sorted(source.glob("*/*/judgements.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"No judgements in {path}")
        indexed = index_by_id(records, str(path))
        model = records[0].get("llm")
        judge = records[0].get("judge_model_name")
        if not isinstance(judge, str) or not judge.strip():
            raise ValueError(f"Every judgement requires judge_model_name: {path}")
        artifacts = score_paths(model, judge, source)
        if path != artifacts["judgements"]:
            raise ValueError(f"Judgement metadata does not match its namespace: {path}")
        if not artifacts["answers"].is_file():
            raise ValueError(f"Source answers are required to establish completeness: {path}")
        answers = [json.loads(line) for line in artifacts["answers"].read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_ids = set(index_by_id(answers, "answers"))
        if set(indexed) - expected_ids:
            raise ValueError(f"Unknown judgement IDs in {path}")
        expected_count = len(expected_ids)
        if artifacts["scores"].is_file():
            previous = json.loads(artifacts["scores"].read_text(encoding="utf-8"))
            prior_count = previous.get("sample_count")
            if type(prior_count) is not int or prior_count != expected_count:
                raise ValueError(f"Conflicting expected sample count in {path}")
        valid = []
        for record in records:
            if record.get("llm") != model or record.get("judge_model_name") != judge:
                raise ValueError(f"Mixed model/judge identities in {path}")
            try:
                valid.append(parse_evaluation(record.get("evaluation")))
            except ValueError:
                continue
        averages = {key: sum(row[key] for row in valid) / len(valid) if valid else None for key in EVALUATION_CATEGORIES}
        summary = {"model_name": model, "judge_model_name": judge, "category_averages": averages,
                   "overall_average": sum(averages.values()) / len(averages) if valid else None,
                   "sample_count": expected_count, "evaluated_count": len(valid), "failed_count": expected_count - len(valid),
                   "status": "complete" if len(valid) == expected_count else "incomplete"}
        write_json(artifacts["scores"], summary)
        count += 1
        failed += summary["failed_count"]
    if failed:
        raise ValueError(f"Reaggregation preserved {failed} invalid or missing judgements; affected summaries are incomplete")
    return count


@click.command()
@click.argument("operation", type=click.Choice(["export", "reaggregate"]))
@click.option("--scores-dir", default="scores", type=click.Path(path_type=Path))
@click.option("--results-dir", default="../results", type=click.Path(path_type=Path))
def main(operation, scores_dir, results_dir):
    """Maintain absolute scores offline without calling inference endpoints."""
    if not scores_dir.is_dir():
        raise click.ClickException(f"Score directory not found: {scores_dir}")
    try:
        count = export_scores(scores_dir, results_dir) if operation == "export" else reaggregate_scores(scores_dir)
    except (ValueError, KeyError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Processed {count} score artifacts")

