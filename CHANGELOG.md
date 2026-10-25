# Changelog

All notable changes to `hermes-plugin-mnemon` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- README: removed the phantom `hermes-plugin-mnemon install` command. The
  package ships no `[project.scripts]` entry, so the command never existed.
  Option C now shows the real post-pip step (locate the installed package
  via `python -c 'import mnemon, ...'` and symlink it into
  `~/.hermes/plugins/`), and the team-sharing note references those steps
  instead of a non-existent installer. See #24.

### Changed
- Store selection no longer mutates the process-global
  `os.environ["MNEMON_STORE"]`. The resolved store is now passed explicitly to
  every `mnemon` CLI invocation via the `--store <name>` flag (available
  since mnemon 0.1.14 — verified across 0.1.14–0.2.12), appended inside
  `_run_mnemon()` as a keyword-only `store=` parameter. Multiple profiles or
  sessions multiplexed in one process can no longer race on a single global
  env var; the store in use is now derivable from each call site.

## [0.1.1] — 2026-10-25

### Fixed
- Use `agent.memory_provider.spawn_context_thread` for all background jobs
  (`queue_prefetch`, `sync_turn`, `on_memory_write`) instead of bare
  `threading.Thread`, per the Hermes memory provider contract.
  A bare thread starts with empty contextvars, which silently lands
  multiplexed profiles on the default profile's store.

### Removed
- Dead `hooks:` list in `mnemon/plugin.yaml`. The memory-provider loader
  does not read this key (it only reads `name`), `on_session_end` is
  unimplemented, and no bundled provider ships a `hooks:` section.

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

[0.1.0]: https://github.com/gitricko/hermes-plugin-mnemon/releases/tag/v0.1.0
