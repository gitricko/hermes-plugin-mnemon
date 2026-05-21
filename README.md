# hermes-plugin-mnemon

[![Python ≥3.11](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/gitricko/hermes-plugin-mnemon/actions/workflows/ci.yml/badge.svg)](https://github.com/gitricko/hermes-plugin-mnemon/actions/workflows/ci.yml)

> **Mnemon graph-based memory for [Hermes Agent](https://github.com/nousresearch/hermes-agent)**
> wraps [`mnemon v0.1.x`](https://github.com/mnemon-dev/mnemon) as a native
> `MemoryProvider`. Gives Hermes a four-graph memory store (temporal, entity,
> causal, semantic) with importance decay, dedup, soft-delete, and optional
> vector recall via Ollama.

For those who want to give this Hermes Agent Memory Provider plugin a spin, head to [Hermes-WebTop](https://github.com/gitricko/hermes-webtop) and start a GitHub Codespace to try it.

---

## Table of Contents

1. [Why this plugin?](#why-this-plugin)
2. [Prerequisites](#prerequisites)
3. [Install / activate](#install--activate)
   - [A. Manual copy (simplest)](#option-a--manual-copy-simplest)
   - [B. Symlink (development)](#option-b--symlink-development)
   - [C. pip install](#option-c--pip-install)
4. [How it works](#how-it-works)
5. [Tools](#tools)
6. [Troubleshooting](#troubleshooting)
7. [Development](#development)
8. [License](#license)

---

## Why this plugin?

Hermes ships with a built-in ~2 kchar session memory — fine for a single
turn, but it flushes on compress and doesn't remember across sessions.
Mnemon adds:

| Feature | Built-in Hermes | Hermes + Mnemon |
|---|---|---|
| Unlimited retention | ✗ | ✓ |
| Importance decay | ✗ | ✓ |
| Semantic / vector recall | ✗ | optional ✓ |
| Cross-session persistence | ✗ | ✓ |
| Multi-agent graph sharing | ✗ | ✓ |
| Soft-delete / forget | ✗ | ✓ |
| Named stores per profile | ✗ | ✓ |

---

## Prerequisites

| Dependency | Version | Install |
|---|---|---|
| [mnemon](https://github.com/mnemon-dev/mnemon) | ≥ 0.1.3 | `curl -fsSL <mnemon install url> \| sh` |
| (optional) [Ollama](https://ollama.ai) | latest | `ollama pull nomic-embed-text` |
| Hermes Agent | ≥ 0.9.0 | see [Hermes docs](https://hermes-agent.nousresearch.com) |
| Python | ≥ 3.11 | system Python or `uv` |

Verify prerequisites:

```bash
mnemon --version   # ≥ 0.1.3
python --version   # ≥ 3.11
```

---

## Install / activate

### Option A — Manual copy (simplest)

```bash
# From this repo root
cp -r mnemon ~/.hermes/plugins/
# Activate in config:
hermes config set memory.provider mnemon
# Or edit ~/.hermes/config.yaml → memory.provider: mnemon
```

### Option B — Symlink (development)

```bash
mkdir -p ~/.hermes/plugins
ln -sfn /path/to/hermes-plugin-mnemon/mnemon ~/.hermes/plugins/mnemon
hermes config set memory.provider mnemon
```

Edits in this repo are live on Hermes restart — no copy step needed.

### Option C — pip install

```bash
pip install hermes-plugin-mnemon
# Then symlink or copy into Hermes' plugin dir:
hermes-plugin-mnemon install   # placeholder — user copies manually
```

> **Note.** Hermes discovers plugins in `~/.hermes/plugins/`, not in `site-packages`.
> A pip install puts the code there but you still need one manual copy or symlink step.
> For teams that share `config.yaml` across machines, add a small post-install script
> (`hermes-plugin-mnemon install`) that performs the symlink automatically.

---

## How it works

### Lifecycle hooks

| Hermes hook | What the plugin does |
|---|---|
| `initialize(session_id, …)` | Derives fixed store from `agent_identity`, ignores `session_id`; creates store if missing and inits ID index |
| `prefetch(query)` | Background `mnemon recall` with intent detection; 30 s TTL cache |
| `sync_turn(user, asst)` | Background `mnemon remember` on every turn |
| `on_memory_write(action, target, content, …)` | Mirrors built-in `memory add/replace` into mnemon |
| `on_pre_compress(messages)` | Persists last 10 key turns before compression |
| `on_session_switch(new_id, …)` | Resets recall cache, updates store env |
| `shutdown()` | Clears recall cache |

### Store naming

```
<profile>    ← fixed per-profile store (survives restarts)
```

Each Hermes profile gets its own dedicated store that is reused across sessions,
so memory persists. Override manually with `MNEMON_STORE=your-name` env var
(takes precedence over the profile‑derived name).

---

## Tools

Three tools are registered on each Hermes session when mnemon is active:

### `mnemon_remember`

```json
{"text": "...", "category": "general", "importance": 3, "entities": [], "tags": []}
```

Store an insight in Mnemon.
Categories: `decision` · `preference` · `fact` · `insight` · `context` · `general`
Importance: 1–5 (auto: `decision` → 4, others → 3)

### `mnemon_recall`

```json
{"query": "...", "intent": "GENERAL", "limit": 8}
```

Recall insights by natural language query.
Intents: `WHY` · `WHEN` · `ENTITY` · `GENERAL`

### `mnemon_forget`

```json
{"insight_id": "uuid-from-recall"}
```

Soft-delete an insight by its ID. Post-delete, it disappears from recall
results only (underlying store rows are untouched — pruned on store rebuild).

---

## Troubleshooting

```bash
# Is mnemon on PATH?
which mnemon && mnemon --version

# Store health
mnemon status

# Hermes reports provider status
hermes memory status

# Manual store smoke-test
MNEMON_STORE=test mnemon remember "hello" --cat fact --imp 3
MNEMON_STORE=test mnemon recall "hello" --limit 3
```

---

## Development

### Run tests (no mnemon binary required)

```bash
pip install ".[dev]"
pytest tests/ -v
```

All mocks are in `tests/test_mnemon.py`. The actual `subprocess.run(["mnemon", …])`
call is patched so the suite passes on any machine — even without mnemon installed.

### Project layout

```
hermes-plugin-mnemon/
├── .github/
│   └── workflows/
│       └── ci.yml           ← ruff + pytest + build on every push
├── tests/
│   ├── __init__.py
│   └── test_mnemon.py       ← 337 lines, 22 tests
├── mnemon/
│   ├── __init__.py          ← full plugin logic (381 lines)
│   └── plugin.yaml          ← plugin metadata for Hermes
├── pyproject.toml
├── README.md
└── LICENSE                  ← MIT
```

---

## License

MIT — see [LICENSE](LICENSE).
© 2026 Nous Research.
