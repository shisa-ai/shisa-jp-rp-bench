# Shisa Japanese RP Bench

Generate Japanese roleplay conversations and score them against an eight-category rubric using an OpenAI-compatible HTTP API. One HTTPX client handles inference; model hosting is external.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The same CLI is available as `japanese-rp-bench` or `python -m japanese_rp_bench`.

## Generate, then judge

From this checkout, export `SHISA_API_KEY` and run the included one-scenario, two-turn example:

```bash
CONVERSATION_FILE=$(japanese-rp-bench generate --config configs/example.yaml)
japanese-rp-bench judge \
  --conversation-file "$CONVERSATION_FILE" \
  --judge-model glm-5.2 \
  --base-url https://api.shisa.ai/openai/v1 \
  --api-key-env SHISA_API_KEY \
  --request-options '{"temperature":0,"max_tokens":8192}' \
  --output-dir results/scores
```

Generation prints the conversation file path. Judging always takes an explicit existing file and uses its embedded model identity and scenario settings. To judge a file without embedded settings, supply `--dataset` with a local JSON/JSONL file or Hugging Face repository. Every conversation must have a consistent nonempty `target_model_name` and a unique scenario `id`.

Edit [configs/example.yaml](configs/example.yaml) for your run. `dataset_repo` accepts a local file or Hugging Face repository; `dataset_split` selects the split. Local JSON accepts a list or split-to-list mapping. The [scenario fixture](configs/smoke_scenario.json) illustrates the required fields.

Configure each participant with `target_` or `user_` fields: `model_name`, `base_url`, `api_key` or `api_key_env`, `timeout`, `max_retries`, and `request_options`. `max_turns`, `max_workers`, and `max_samples` bound the workload. The generate command can override workers/samples with flags. Configure token limits explicitly using `max_tokens` or `max_completion_tokens` in request options; generation defaults to 1024 if neither is supplied.

Explicit keys take precedence over named environment variables. Without explicit settings, endpoint fallback is `OPENAI_COMPATIBLE_API_URL`, then `OPENAI_BASE_URL`, then `https://api.openai.com/v1`; key fallback is `OPENAI_COMPATIBLE_API_KEY`, then `OPENAI_API_KEY`. `SHISA_API_KEY` is a fallback only for `api.shisa.ai`. Keyless local endpoints are supported.

## Batch and saved scores

To judge a directory containing one conversation file per target model:

```bash
japanese-rp-bench batch \
  --conversations-dir results/conversations \
  --judge-model glm-5.2 \
  --base-url https://api.shisa.ai/openai/v1 \
  --api-key-env SHISA_API_KEY \
  --output-dir results/scores
```

Batch rankings include only fresh, complete results matching the requested model, judge, and sample counts. The CSV is written as `model_rankings_<escaped-judge>.csv` in the working directory. Separate different runs of the same model into different batch input directories.

Score maintenance runs without model calls:

```bash
japanese-rp-bench scores export --scores-dir results/scores --results-dir results/export
japanese-rp-bench scores reaggregate --scores-dir results/scores
```

## Output and failures

Generated files live under `<output_dir>/conversations/`, with a hash of nonsecret generation settings in their names. Completed conversations are checkpointed as they finish, in source order; failures are recorded separately under `generation_failures/`.

Judging writes `<output_dir>/<escaped-target>/<escaped-judge>/{scores.json,judgements.jsonl,answers.jsonl}`. All eight category scores must be integers from 1 to 5 with a nonempty reason. Invalid judgments retain raw output or sanitized errors and produce an incomplete summary. Failed generation, incomplete judging, or failed batch evaluation returns a nonzero exit status. Transport retries are bounded.

Writes use atomic replacement. Repeating the same run in the same output directory replaces its artifacts; use separate directories for experiments or concurrent runs. Partial work is preserved, but automatic resume is not implemented. The full default rubric ships in the package; `judge --prompt-file` selects a custom rubric.

## Development

```bash
python -m pytest
python -m pytest --cov --cov-report=term-missing --cov-fail-under=95
```

Tests run offline without credentials or model downloads. See [maintenance notes](docs/MAINTENANCE.md) for verification, migration, and retained historical evidence. Use `japanese-rp-bench COMMAND --help` for command options.
