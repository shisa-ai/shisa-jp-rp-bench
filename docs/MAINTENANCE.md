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

**Thinking support verification (2026-10-05): 373 tests passed, zero failures or expected failures, with 97.13% combined statement/branch coverage.** Tests cover separate reasoning/final answers, bounded token-budget retries, provider controls, truncated-response rejection, concurrent calls, CLI forwarding, and diagnostics retained after later-turn failures. The rebuilt wheel passed new judge/batch CLI option checks, reasoning extraction and budget retry checks, and a packaged-rubric check outside the checkout. Independent reviews found later-turn diagnostics loss and missing token accounting for malformed responses; regression tests cover both fixes.

Two tiny live Shisa requests, one each to `qwen3.8-27b` and `glm-5.2`, returned complete final answers with reasoning stored separately and backend-default reasoning preserved. These are integration checks, not a new benchmark. The local artifact is `results/thinking-support-smoke-2026-10-05.json`; existing preliminary rankings were not rewritten.

The preceding cleanup verification passed 322 tests with 96.96% coverage, all CLI help entrypoints, installed-wheel batch/score commands, packaged-rubric checks, `pip check`, archive hashes, and `git diff --check`. That cleanup verification was offline.

## Historical evidence

Previous reviews, implementation plans, validation reports, historical datasets/results/prompts, and live-test output were preserved outside the active checkout at the sibling directory `../shisa-jp-rp-bench-history-2026-10-05/`. Its `preservation-manifest.json` records paths, sizes, and SHA-256 hashes; all 28 source files were verified after moving. This local history directory is not part of the repository.

The earlier live integration used `shisa-ai/shisa-v2.1-llama3.3-70b` to generate one two-turn scenario and `glm-5.2` to judge it on `https://api.shisa.ai/openai/v1`. One of one judgments passed the schema with zero failures. It was an integration test, not a statistical benchmark. Credentials were not retained.
