# Contributing

Issues and pull requests are welcome. Please discuss broad changes in an issue first.

1. Fork `https://github.com/GunnerJyo/aws-ai-engineering-toolkit` and clone your fork.
2. Create a virtual environment and install `python -m pip install -e ".[dev]"`.
3. Create a focused branch: `git switch -c feat/short-description`.
4. Add or update offline tests; do not make the default suite call AWS.
5. Run `python -m pytest`, `ruff check .`, and `ruff format --check .`.
6. Update documentation where behavior or commands change.
7. Push your branch to your fork and open a pull request against `main`.

Use conventional, descriptive commit messages. Do not include credentials, customer data, or generated build artifacts. The project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Good first issues

- Add input-size boundary tests for Titan embedding requests.
- Add more examples using deterministic local embedder test doubles.
- Improve troubleshooting notes for common region/model-access errors.
- Add an optional persistent vector-store adapter behind a small interface.
