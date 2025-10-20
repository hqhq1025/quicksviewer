# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `quicksviewer/`, with task-specific subpackages such as `model/` for network architectures, `train/` for training loops and collators, `serve/` for CLI and Gradio interfaces, and `eval/` for benchmark harnesses. Shared helpers sit in `utils/` and preprocessing utilities in `preprocess/`. Training launchers and DeepSpeed configs live in `scripts/`, while `docs/` holds narrative guides, `playground/` stores sample media, and checkpoints or artifacts should land under `output/`. Mirror this layout when adding new modules or experiments.

## Build, Test, and Development Commands
- `conda create -n quicksviewer python=3.11 -y`: create the reference development environment.
- `pip install -r requirements.txt && pip install -e .`: install runtime and editable package dependencies.
- `bash scripts/stage1.sh` (and `stage2.sh`, `stage3.sh`): reproduce the staged training pipeline, writing checkpoints to `output/`.
- `bash quicksviewer/eval/run_eval.sh videomme checkpoints/quicksviewer-s3/checkpoint-10000 420 1`: evaluate a specific checkpoint on VideoMME.
- `PYTHONPATH=/path/to/quicksviewer python quicksviewer/serve/cli.py --model-path <checkpoint> --context <media>`: run local inference against a saved model.

## Coding Style & Naming Conventions
- Target Python 3.11 features and standard library types; prefer explicit type hints.
- Format Python with `black` (configured for 240-character lines) and four-space indentation.
- Use snake_case for functions and modules, PascalCase for classes, and reserve UPPER_CASE for constants defined in `constants.py`.
- Keep imports intra-package (e.g., `from quicksviewer.train import datamodule`) and place related dataclasses alongside their usage.

## Testing Guidelines
Pytest is bundled with the training extras. Add unit tests mirroring the package layout (e.g., `tests/test_train/test_video_sampler.py`) and name them descriptively such as `test_select_sparse_frames`. Run the full suite with `pytest`. When adding new training stages or evaluation flows, capture metrics JSON or console logs and attach them to your PR as functional evidence.

## Commit & Pull Request Guidelines
- Write commit subjects in concise imperative form (e.g., `update ckpts for stage3`) and group related changes together.
- Reference datasets, checkpoints, or experiment IDs in the commit body when relevant.
- Open PRs with a short intent summary, runnable commands, linked tracking issues, and before/after metrics or screenshots for UI-facing updates.
- Note any documentation, config, or sample updates required to reproduce results.

## Security & Configuration Tips
- Never commit API keys, proprietary datasets, or checkpoints; rely on `.gitignore` and secure storage.
- Validate new DeepSpeed or training configs against the templates in `scripts/` before sharing.
- Document any required environment variables or secrets in `docs/` so contributors can recreate the setup safely.
