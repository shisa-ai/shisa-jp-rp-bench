"""Run absolute judges and rank only fresh, complete results from this batch."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import click

from .artifacts import safe_name, score_paths
from .utils import EVALUATION_CATEGORIES


def load_score_summary(path, model, judge, expected_count):
    """Validate the standalone judge's artifact before ranking or publishing it."""
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or result.get("model_name") != model or result.get("judge_model_name") != judge:
        raise ValueError("Score artifact model/judge does not match this evaluation")
    if result.get("status") != "complete" or result.get("failed_count") != 0:
        raise ValueError("Score artifact is incomplete")
    if result.get("sample_count") != expected_count or result.get("evaluated_count") != expected_count:
        raise ValueError("Score artifact sample counts do not match this evaluation")
    averages = result.get("category_averages")
    if not isinstance(averages, dict) or set(averages) != set(EVALUATION_CATEGORIES):
        raise ValueError("Score artifact must contain all rubric categories")
    for value in [*averages.values(), result.get("overall_average")]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 5:
            raise ValueError("Score artifact contains invalid averages")
    expected_average = sum(averages.values()) / len(averages)
    if not math.isclose(result["overall_average"], expected_average, rel_tol=1e-9):
        raise ValueError("Overall score differs from category average")
    return {"Model": model, "Overall Score": result["overall_average"], **averages}


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@click.command()
@click.option('--judge-model', required=True)
@click.option('--max-samples', default=20, type=click.IntRange(min=1))
@click.option('--conversations-dir', default='conversations', type=click.Path(path_type=Path))
@click.option('--output-dir', default='scores', type=click.Path(path_type=Path))
@click.option('--base-url', help='OpenAI-compatible judge endpoint.')
@click.option('--api-key-env', help='Environment variable containing the judge API key.')
@click.option('--dataset', help='Scenario dataset repository or local JSON/JSONL file.')
def main(judge_model, max_samples, conversations_dir, output_dir, base_url, api_key_env, dataset):
    """Evaluate each conversation file with HTTPX and rank this batch's results."""
    if not conversations_dir.is_dir():
        raise click.ClickException(f"{conversations_dir} directory not found")
    files = sorted(conversations_dir.glob('*.jsonl'))
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    seen_models = set()
    try:
        for source in files:
            with source.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            if not records:
                raise ValueError(f"Conversation file is empty: {source}")
            model = records[0].get("target_model_name")
            if not isinstance(model, str) or not model.strip() or model in seen_models:
                raise ValueError(f"Invalid or duplicate model identity in {source}")
            if any(row.get("target_model_name") != model for row in records):
                raise ValueError(f"Mixed model identities in {source}")
            seen_models.add(model)
            jobs.append((source, model, min(max_samples, len(records))))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise click.ClickException(str(exc)) from exc

    results = []
    stage = Path(tempfile.mkdtemp(prefix=".batch-", dir=output_dir)).resolve()
    try:
        for source, model, count in jobs:
            command = [sys.executable, '-m', 'japanese_rp_bench', 'judge',
                       '--conversation-file', str(source.resolve()),
                       '--judge-model', judge_model, '--max-samples', str(count),
                       '--output-dir', str(stage)]
            for flag, value in (("--base-url", base_url), ("--api-key-env", api_key_env), ("--dataset", dataset)):
                if value is not None:
                    command.extend([flag, value])
            click.echo(f"Evaluating {model} with {judge_model}")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise ValueError(f"Evaluation failed for {model} (exit {completed.returncode})")
            current = score_paths(model, judge_model, stage)
            summary = load_score_summary(current['scores'], model, judge_model, count)
            if not all(path.is_file() for path in current.values()):
                raise ValueError(f"Evaluation artifacts missing for {model}")
            destination = score_paths(model, judge_model, output_dir)
            for key, path in current.items():
                atomic_copy(path, destination[key])
            results.append(summary)
        results.sort(key=lambda row: (-row['Overall Score'], row['Model']))
        columns = ['Model', 'Overall Score'] + (list(EVALUATION_CATEGORIES) if results else [])
        output_file = Path(f"model_rankings_{safe_name(judge_model)}.csv")
        csv_stage = stage / "rankings.csv"
        with csv_stage.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        atomic_copy(csv_stage, output_file)
        click.echo(f"Rankings ({judge_model}):")
        for rank, result in enumerate(results, start=1):
            model_label = json.dumps(result["Model"], ensure_ascii=False)
            click.echo(f"{rank}. {model_label}: {result['Overall Score']:.2f}")
        if not results:
            click.echo("No conversation files found; saved an empty ranking table.")
        click.echo(f"Rankings saved to {output_file}")
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"{exc}. Batch artifacts retained at {stage}") from exc
    else:
        shutil.rmtree(stage)

