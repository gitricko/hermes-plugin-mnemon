# Changelog

All notable changes to `hermes-plugin-mnemon` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-20

### Added
- Initial release of `MnemonMemoryProvider` wrapping `mnemon CLI v0.1.3`
- `mnemon_remember` / `mnemon_recall` / `mnemon_forget` tool schema + dispatch
- `prefetch()` with intent detection and 30 s TTL recall cache
- `sync_turn()` background auto-remember on every conversation turn
- `on_memory_write()` mirror hook: sinks built-in Hermes memory writes into mnemon
- `on_pre_compress()` persists last 10 key turns before session compression
- Postgres-style sidecar ID index at `~/.hermes/mnemon_id_index.json`
- Store isolation per profile + session via `MNEMON_STORE` env var
- pip packaging via `pyproject.toml` (`hermes-plugin-mnemon`)

### dev
- 22 unit tests covering all hooks, tool calls, parsers and intent detection
- GitHub Actions CI: ruff → pytest → build

[0.1.0]: https://github.com/mnemon-dev/hermes-plugin-mnemon/releases/tag/v0.1.0
