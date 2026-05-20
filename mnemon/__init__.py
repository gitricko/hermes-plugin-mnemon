"""
Mnemon memory plugin — Hermes MemoryProvider wrapping the `mnemon` CLI.

Graph-based LLM-supervised memory with four-graph store (temporal, entity,
causal, semantic). Requires the `mnemon` binary on PATH (mnemon v0.1.x).

Install:
  hermes config set memory.provider mnemon         (activate in Hermes)
  # or edit ~/.hermes/config.yaml → memory.provider: mnemon

Docs: https://github.com/mnemon-dev/mnemon
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level mnemon CLI helpers
# ---------------------------------------------------------------------------

def _run_mnemon(cmd_args: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        r = subprocess.run(["mnemon"] + cmd_args,
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", "mnemon: not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", "mnemon: timeout"
    except Exception as exc:
        return 1, "", str(exc)


def _json_output(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    try:
        for line in reversed(text.strip().splitlines()):
            try:
                return json.loads(line)
            except Exception:
                continue
    except Exception:
        pass
    return {"raw": text.strip()}


def _fmt_hits(hits: list[dict]) -> str:
    if not hits:
        return ""
    parts = []
    for h in hits:
        insight = h.get("insight", h)
        text = insight.get("content", h.get("text", str(insight)))
        score = h.get("score")
        cat = insight.get("category", "general")
        if score is not None:
            parts.append(f"- [{cat}] ({score:.2f}) {text}")
        else:
            parts.append(f"- [{cat}] {text}")
    return "\n".join(["[mnemon recall]"] + parts + [""])


_FORGET_RE = re.compile(r"forget|remove|delete|erase|clear|wipe", re.I)
_IDX_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# MemoryProvider
# ---------------------------------------------------------------------------

class MnemonMemoryProvider(MemoryProvider):
    name = "mnemon"

    def __init__(self):
        self._store: str | None = None
        self._session_id: str = ""
        self._index_path: Path = Path.home() / ".hermes" / "mnemon_id_index.json"
        self._recall_cache: list | None = None
        self._last_prefetch_at: float = 0.0
        self._prefetch_ttl: int = 30

    INTENT_MAP = {
        "why": "WHY", "because": "WHY",
        "when": "WHEN", "how long": "WHEN",
        "who": "ENTITY", "what is": "ENTITY", "entity": "ENTITY",
    }

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower().strip()
        for pattern, intent in MnemonMemoryProvider.INTENT_MAP.items():
            if pattern in q:
                return intent
        return "GENERAL"

    # ------------------------------------------------------------------ #
    # Core lifecycle
    # ------------------------------------------------------------------ #

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "store",
                "description": "Mnemon memory store name (defaults to profile name)",
                "required": False,
                "default": ""
            }
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        if not hermes_home:
            return
        config_file = Path(hermes_home) / "mnemon.json"
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(json.dumps(values, indent=2))
        except Exception as e:
            logger.error("Failed to save mnemon config: %s", e)

    def is_available(self) -> bool:
        code, _, _ = _run_mnemon(["--version"])
        return code == 0

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            self._index_path = Path(hermes_home) / "mnemon_id_index.json"

        # Load store config if it exists
        store_config = None
        if hermes_home:
            config_file = Path(hermes_home) / "mnemon.json"
            if config_file.exists():
                try:
                    config_data = json.loads(config_file.read_text())
                    store_config = config_data.get("store")
                except Exception as e:
                    logger.warning("Failed to load mnemon.json: %s", e)

        profile = kwargs.get("agent_identity", "default")
        # Precedence: config value -> environment variable -> fallback to profile
        self._store = store_config or os.environ.get("MNEMON_STORE") or profile

        # create store if needed (rc=0 created, rc=1 exists → both fine)
        code, _, _ = _run_mnemon(["store", "create", self._store], timeout=10)
        if code not in (0, 1):
            logger.warning("mnemon store create rc=%d", code)
        os.environ["MNEMON_STORE"] = self._store
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("mnemon: store=%s profile=%s index=%s", self._store, profile, self._index_path)

    def system_prompt_block(self) -> str:
        return "\n[mongraph]\nMnemon graph-memory is active. Context is fetched before each turn.\n"

    # ---------- prefetch ------------------------------------------------ #

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        now = time.time()
        if self._recall_cache and (now - self._last_prefetch_at) < self._prefetch_ttl:
            return _fmt_hits(self._recall_cache)
        intent = self._detect_intent(query)
        code, stdout, stderr = _run_mnemon(
            ["recall", query, "--limit", "8", "--intent", intent], timeout=10,
        )
        if code != 0:
            logger.debug("mnemon recall error: %s", stderr.strip())
            return ""
        data = _json_output(stdout)
        hits = data.get("results", [])
        if isinstance(data, dict) and not hits:
            logger.debug("mnemon recall empty: %s", stdout[:200])
            return ""
        self._recall_cache = hits
        self._last_prefetch_at = now
        return _fmt_hits(hits)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        threading.Thread(target=self.prefetch,
                         args=(query,), kwargs={"session_id": session_id},
                         daemon=True).start()

    # ---------- sync_turn ----------------------------------------------- #

    @staticmethod
    def _auto_cat(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["choose", "decided", "picked", "selected", "chose"]):
            return "decision"
        return "general"

    def _auto_remember(self, user_text: str, asst_text: str) -> None:
        combined = f"{user_text} {asst_text}"
        if _FORGET_RE.search(combined):
            return
        for text in (user_text.strip(), asst_text.strip()):
            if not text or len(text) < 30:
                continue
            cat = self._auto_cat(text)
            imp = 4 if cat == "decision" else 3
            self._remember_and_index(text, category=cat, importance=imp)

    def _remember(self, text: str, category: str = "general",
                  importance: int = 3,
                  entities: list[str] | None = None,
                  tags: list[str] | None = None,
                  source: str = "agent") -> str | None:
        args = ["remember", text, "--cat", category,
                "--imp", str(importance), "--source", source]
        if entities:
            args += ["--entities", ",".join(entities)]
        if tags:
            args += ["--tags", ",".join(tags)]
        code, stdout, _ = _run_mnemon(args, timeout=15)
        if code != 0:
            return None
        return _json_output(stdout).get("id")

    def _remember_and_index(self, text: str, **kwargs) -> str | None:
        iid = self._remember(text, **kwargs)
        if not iid:
            return None
        with _IDX_LOCK:
            idx: dict[str, Any] = {}
            if self._index_path.exists():
                try:
                    idx = json.loads(self._index_path.read_text())
                except Exception:
                    idx = {}
            idx.setdefault("ids", {})
            idx["ids"][iid] = {
                "ts": datetime.now(datetime.UTC).isoformat(),
                "text_snippet": text[:80],
                "store": self._store,
                "session": self._session_id,
            }

            # Prevent the index from growing indefinitely (cap at 2000 recent items)
            if len(idx["ids"]) > 2000:
                # Remove oldest keys (first inserted) to keep the most recent 2000
                old_keys = list(idx["ids"].keys())[:-2000]
                for key in old_keys:
                    idx["ids"].pop(key, None)

            self._index_path.write_text(json.dumps(idx, indent=2))
        return iid

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        threading.Thread(target=self._auto_remember,
                         args=(user_content, assistant_content),
                         daemon=True).start()

    # ---------- session switch ------------------------------------------ #

    def on_session_switch(
        self, new_session_id: str, *,
        parent_session_id: str = "", reset: bool = False, **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id
        self._recall_cache = None
        self._last_prefetch_at = 0
        if reset:
            self._store = None
        if self._store:
            os.environ["MNEMON_STORE"] = self._store

    # ---------- pre-compress ------------------------------------------- #

    def on_pre_compress(self, messages: list[dict]) -> str:
        key_msgs = [m for m in messages if m.get("role") in ("assistant","user")
                    and len(m.get("content","")) > 80][-10:]
        for msg in key_msgs:
            self._remember_and_index(msg["content"][: 600],
                                     category="context", importance=2)
        return ""

    def shutdown(self) -> None:
        self._recall_cache = None

    # ---------- mirror built-in memory writes -------------------------- #

    def on_memory_write(self, action: str, target: str,
                        content: str, metadata: dict | None = None) -> None:
        cat = "preference" if target == "user" else "general"
        imp = 4 if cat == "preference" else 3
        src = (metadata or {}).get("write_origin", "agent")
        threading.Thread(
            target=lambda: self._remember_and_index(content[:2500],
                                                     category=cat,
                                                     importance=imp,
                                                     source=src),
            daemon=True,
        ).start()

    # ---------- tool schemas + dispatch -------------------------------- #

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "mnemon_remember",
                "description": "Store an insight in Mnemon graph memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Insight text"},
                        "category": {"type": "string",
                                     "enum": ["decision","preference","fact","insight","context","general"],
                                     "default": "general"},
                        "importance": {"type": "integer", "minimum": 1, "maximum": 5,
                                       "default": 3},
                        "entities": {"type": "array", "items": {"type":"string"},
                                     "default": []},
                        "tags": {"type": "array", "items": {"type":"string"},
                                 "default": []},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "mnemon_recall",
                "description": "Recall insights from Mnemon by natural-language query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look up"},
                        "intent": {"type": "string",
                                   "enum": ["WHY","WHEN","ENTITY","GENERAL"],
                                   "default": "GENERAL"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20,
                                  "default": 8},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mnemon_forget",
                "description": "Soft-delete an insight in Mnemon by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "insight_id": {"type": "string",
                                       "description": "Insight ID from recall"},
                    },
                    "required": ["insight_id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "mnemon_remember":
            text = args.get("text", "").strip()
            if not text:
                return tool_error("text is required")
            iid = self._remember_and_index(
                text,
                category=args.get("category", "general"),
                importance=args.get("importance", 3),
                entities=args.get("entities", []),
                tags=args.get("tags", []),
                source=args.get("source", "agent"),
            )
            return json.dumps({"success": bool(iid), "id": iid or "error"})

        if tool_name == "mnemon_recall":
            q = args.get("query", "").strip()
            if not q:
                return tool_error("query is required")
            code, stdout, stderr = _run_mnemon(
                ["recall", q,
                 "--limit", str(args.get("limit", 8)),
                 "--intent", args.get("intent", "GENERAL")],
                timeout=15,
            )
            if code != 0:
                return tool_error(f"recall error ({code}): {stderr.strip()}")
            data = _json_output(stdout)
            hits = data.get("results", [])
            return json.dumps({"hits": hits, "count": len(hits)}, indent=2)

        if tool_name == "mnemon_forget":
            iid = args.get("insight_id", "").strip()
            if not iid:
                return tool_error("insight_id is required")
            code, stdout, stderr = _run_mnemon(["forget", iid], timeout=10)
            if code != 0:
                return tool_error(f"forget error ({code}): {stderr.strip()}")
            with _IDX_LOCK:
                if self._index_path.exists():
                    try:
                        idx = json.loads(self._index_path.read_text())
                        idx.get("ids", {}).pop(iid, None)
                        self._index_path.write_text(
                            json.dumps(idx, indent=2))
                    except Exception:
                        pass
            return json.dumps({"success": True, "forgotten": iid})

        return tool_error(f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# Hermes plugin registration
# ---------------------------------------------------------------------------

def register(ctx):
    ctx.register_memory_provider(MnemonMemoryProvider())
