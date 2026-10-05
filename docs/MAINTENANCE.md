# Maintenance

The supported workflow is `generate` → `judge` or `batch`, with `scores export|reaggregate` for saved results. The package contains all executable code; there are no shell wrappers, root-level Python launchers, local model servers, Slurm jobs, pairwise-ranking tools, or dataset-upload tools.

## Migration

- Invoke the installed `japanese-rp-bench` CLI or `python -m japanese_rp_bench`.
- Generation does not judge. Remove `no_judge`, all `judge_*` generation config, `evaluation_prompt_file`, inference-method fields, and tensor-parallel settings. Unknown generation keys are errors.
- Remove `low_context`/`ultra_low_context` flags and `JP_RP_MAX_TOKENS`. Set token limits in each participant's request options.
- Judging requires `--conversation-file` and `--judge-model`. Old aliases and filename/model discovery are removed. Conversation records must carry their target model identity. Supply `--dataset` explicitly if scenario settings are absent.
- Python client creation is `create_client(...) -> ChatClient`. Generation is `japanese_rp_bench.run.generate_conversations(config) -> Path`. All scoring uses `japanese_rp_bench.judging`; there is no separate integrated evaluator.
- Keep separate output directories for experiments. Checkpoints preserve partial work, but do not implement automatic resume.

## Dependencies and tests

Direct runtime dependencies are HTTPX, datasets, Click, and PyYAML. Dataset loading may bring NumPy/pandas transitively. No provider SDK, inference runtime, or ranking engine is required.

Tests use HTTPX MockTransport, real temporary files, and actual CLI/artifact processing. An offline socket guard fails a test even when application code catches the attempted connection. CI tests Python 3.10 and 3.13 with a 95% combined statement/branch coverage floor.

**Verification (2026-10-05): 322 tests passed, zero failures or expected failures, with 96.96% combined statement/branch coverage.** The installed wheel passed all CLI help entrypoints, batch/score commands outside the checkout, packaged-rubric checks, and `pip check`. Independent review found no additional actionable interface defects. The archive hashes and `git diff --check` also passed. No new live model requests were made.

## Historical evidence

Previous reviews, implementation plans, validation reports, historical datasets/results/prompts, and live-test output were preserved outside the active checkout at the sibling directory `../shisa-jp-rp-bench-history-2026-10-05/`. Its `preservation-manifest.json` records paths, sizes, and SHA-256 hashes; all 28 source files were verified after moving. This local history directory is not part of the repository.

The earlier live integration used `shisa-ai/shisa-v2.1-llama3.3-70b` to generate one two-turn scenario and `glm-5.2` to judge it on `https://api.shisa.ai/openai/v1`. One of one judgments passed the schema with zero failures. It was an integration test, not a statistical benchmark. Credentials were not retained. The current cleanup is verified offline; no new paid requests or dataset uploads are needed.
