"""
Unit tests for hermes-plugin-mnemon.

All subprocess calls to the `mnemon` binary are patched so tests run
without mnemon being installed.  Hermes-specific imports are stubbed so
the plugin can be imported and tested in isolation.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out Hermes modules so `mnemon.__init__` can be imported
# ---------------------------------------------------------------------------

agent_mod = types.ModuleType("agent")
agent_mem_provider = types.ModuleType("agent.memory_provider")

class _MemoryProvider:
    """Minimal stand-in for Hermes' MemoryProvider ABC."""
    name = ""


import threading as _stub_threading  # noqa: E402


def _stub_spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
    """Stub mirroring the real agent.memory_provider.spawn_context_thread.

    Returns an unstarted threading.Thread so callers can assert on the
    arguments and call .start() exactly as the real one expects.
    """
    return _stub_threading.Thread(
        target=target, args=args, kwargs=kwargs, name=name, daemon=daemon,
    )


agent_mem_provider.MemoryProvider = _MemoryProvider
agent_mem_provider.spawn_context_thread = _stub_spawn_context_thread
sys.modules["agent"] = agent_mod
sys.modules["agent.memory_provider"] = agent_mem_provider

tools_mod = types.ModuleType("tools")
tools_reg_mod = types.ModuleType("tools.registry")
tools_reg_mod.tool_error = lambda msg: json.dumps({"error": msg})
sys.modules["tools"] = tools_mod
sys.modules["tools.registry"] = tools_reg_mod

# Add the package root to path so `mnemon` can be imported
PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from mnemon import MnemonMemoryProvider  # noqa: E402  (after stubs)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MNEMON_OK = json.dumps({"id": "test-id-123", "content": "stored", "importance": 3})
RECALL_OK = json.dumps({
    "results": [
        {"insight": {"id": "i1", "content": "Hermes uses memory", "category": "fact"}, "score": 0.87},
        {"insight": {"id": "i2", "content": "mnemon is the provider", "category": "insight"}, "score": 0.62},
    ]
})
FORGET_OK = json.dumps({"success": True, "forgotten": "i1"})


class MockResult:
    def __init__(self, rc=0, out="", err=""):
        self.returncode = rc
        self.stdout = out
        self.stderr = err


# ---------------------------------------------------------------------------
# Tests: _run_mnemon helpers
# ---------------------------------------------------------------------------

class TestRunMnemon(unittest.TestCase):

    @patch("mnemon.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MockResult(0, MNEMON_OK, "")
        # We call the private helper via module
        from mnemon import _run_mnemon
        rc, out, err = _run_mnemon(["remember", "hello"])
        mock_run.assert_called_once_with(
            ["mnemon", "remember", "hello"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(rc, 0)

    @patch("mnemon.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="mnemon", timeout=5)
        from mnemon import _run_mnemon
        rc, out, err = _run_mnemon(["recall", "x"])
        self.assertEqual(rc, 124)
        self.assertIn("timeout", err)

    @patch("mnemon.subprocess.run")
    def test_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        from mnemon import _run_mnemon
        rc, out, err = _run_mnemon(["--version"])
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)

    @patch("mnemon.subprocess.run")
    def test_store_flag_appended_to_argv(self, mock_run):
        mock_run.return_value = MockResult(0, MNEMON_OK, "")
        from mnemon import _run_mnemon
        rc, out, err = _run_mnemon(["recall", "x"], store="my-store")
        mock_run.assert_called_once_with(
            ["mnemon", "recall", "x", "--store", "my-store"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(rc, 0)

    @patch("mnemon.subprocess.run")
    def test_store_flag_omitted_when_none(self, mock_run):
        mock_run.return_value = MockResult(0, MNEMON_OK, "")
        from mnemon import _run_mnemon
        rc, out, err = _run_mnemon(["recall", "x"], store=None)
        mock_run.assert_called_once_with(
            ["mnemon", "recall", "x"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(rc, 0)


class TestJsonOutput(unittest.TestCase):

    def test_clean_json(self):
        from mnemon import _json_output
        result = _json_output(MNEMON_OK)
        self.assertIn("id", result)

    def test_malformed_fallback(self):
        from mnemon import _json_output
        result = _json_output("not json at all\nstill not json")
        self.assertIn("raw", result)

    def test_last_line_json(self):
        from mnemon import _json_output
        result = _json_output("header noise\n" + RECALL_OK)
        self.assertIn("results", result)


class TestFmtHits(unittest.TestCase):

    def test_empty(self):
        from mnemon import _fmt_hits
        self.assertEqual(_fmt_hits([]), "")

    def test_single_hit_with_score(self):
        from mnemon import _fmt_hits
        hits = [{"insight": {"content": "hello", "category": "fact"}, "score": 0.9}]
        out = _fmt_hits(hits)
        self.assertIn("[mnemon recall]", out)
        self.assertIn("(0.90)", out)

    def test_hit_without_score(self):
        from mnemon import _fmt_hits
        hits = [{"insight": {"content": "test", "category": "general"}}]
        out = _fmt_hits(hits)
        self.assertNotIn("None", out)


# ---------------------------------------------------------------------------
# Tests: MnemonMemoryProvider lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle(unittest.TestCase):

    def _make_provider(self, mock_available=True):
        p = MnemonMemoryProvider()
        if mock_available:
            with patch("mnemon._run_mnemon") as m:
                m.return_value = (0, "mnemon 0.1.3", "")
                self.assertTrue(p.is_available())
        return p

    @patch("mnemon._run_mnemon")
    def test_initialize_creates_store(self, mock_run):
        mock_run.return_value = (0, "", "")
        p = self._make_provider(mock_available=False)
        with patch("mnemon._run_mnemon") as m:
            m.return_value = (0, "", "")
            p.initialize("ses-123", agent_identity="default")
        self.environ = unittest.mock.patch.dict("os.environ", {}, clear=True)
        self.addCleanup(self.environ.__exit__)
        self.environ.__enter__()
        p.initialize("ses-123", agent_identity="default")

    @patch("mnemon._run_mnemon")
    def test_initialize_does_not_mutate_environ(self, mock_run):
        mock_run.return_value = (0, "", "")
        p = MnemonMemoryProvider()
        sentinel = "SENTINEL-STORE-DO-NOT-TOUCH"
        with patch.dict("os.environ", {
            "MNEMON_STORE": sentinel,
        }, clear=False):
            p.initialize("ses-abc", agent_identity="bot")
            # New contract: the store is forwarded per call on the CLI argv,
            # never written to the process-global environ.
            self.assertEqual(os.environ["MNEMON_STORE"], sentinel)
        # Store resolves to the env value (precedence rule), forwarded per call.
        for call in mock_run.call_args_list:
            self.assertIn("store", call[1])
            self.assertEqual(call[1]["store"], sentinel)

    @patch("mnemon._run_mnemon")
    def test_initialize_passes_store_kwarg(self, mock_run):
        mock_run.return_value = (0, "", "")
        p = MnemonMemoryProvider()
        with patch.dict("os.environ", {"MNEMON_STORE": "default"}, clear=False):
            p.initialize("ses-abc")
            self.assertEqual(p._store, "default")
        for call in mock_run.call_args_list:
            self.assertEqual(call[1].get("store"), "default")

    @patch("mnemon._run_mnemon")
    def test_shutdown(self, mock_run):
        p = MnemonMemoryProvider()
        p._recall_cache = [1, 2, 3]
        p.shutdown()
        self.assertIsNone(p._recall_cache)

    def test_system_prompt_block(self):
        p = MnemonMemoryProvider()
        block = p.system_prompt_block()
        self.assertIn("mnemon", block.lower())


# ---------------------------------------------------------------------------
# Tests: pre-fetch / recall cache
# ---------------------------------------------------------------------------

class TestPrefetch(unittest.TestCase):

    @patch("mnemon._run_mnemon")
    def test_cache_ttl(self, mock_run):
        mock_run.return_value = (0, RECALL_OK, "")
        p = MnemonMemoryProvider()
        p._store = "default"
        out1 = p.prefetch("how do I debug?")
        out2 = p.prefetch("how do I debug?")  # should hit cache
        mock_run.assert_called_once()      # second call is cached
        self.assertIn("store", mock_run.call_args[1])
        self.assertEqual(mock_run.call_args[1]["store"], "default")

    @patch("mnemon._run_mnemon")
    def test_empty_recall(self, mock_run):
        mock_run.return_value = (0, '{"results":[]}', "")
        p = MnemonMemoryProvider()
        p._store = "default"
        out = p.prefetch("obscure query xyz")
        self.assertEqual(out, "")
        self.assertEqual(mock_run.call_args[1].get("store"), "default")

    @patch("mnemon._run_mnemon")
    def test_intent_detection(self, mock_run):
        mock_run.return_value = (0, RECALL_OK, "")
        p = MnemonMemoryProvider()
        p._store = "default"
        p.prefetch("why did it fail?", session_id="s1")
        called_args = mock_run.call_args[0][0]
        self.assertIn("WHY", called_args)
        self.assertIn("recall", called_args)
        self.assertEqual(mock_run.call_args[1].get("store"), "default")


# ---------------------------------------------------------------------------
# Tests: remember / recall / forget tools
# ---------------------------------------------------------------------------

class TestToolCalls(unittest.TestCase):

    def setUp(self):
        self.p = MnemonMemoryProvider()
        self.p._store = "test-store"
        self.p._session_id = "s1"

    @patch("mnemon._run_mnemon")
    def test_remember_returns_id(self, mock_run):
        mock_run.return_value = (0, MNEMON_OK, "")
        # patch index write
        with patch.object(self.p, "_remember_and_index") as mock_idx:
            mock_idx.return_value = "idx-1"
            result = self.p.handle_tool_call("mnemon_remember", {"text": "foo"})
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertEqual(data["id"], "idx-1")

    def test_remember_empty_text(self):
        result = self.p.handle_tool_call("mnemon_remember", {"text": ""})
        data = json.loads(result)
        self.assertIn("error", data)

    @patch("mnemon._run_mnemon")
    def test_remember_error_reports_stderr(self, mock_run):
        mock_run.return_value = (1, "", "content too long (10000 chars, max 8000)")
        idx_path = Path("/tmp/test_mnemon_index_%d.json" % id(self))
        with patch.object(self.p, "_index_path", idx_path):
            result = self.p.handle_tool_call("mnemon_remember", {"text": "foo"})
        data = json.loads(result)
        self.assertFalse(data["success"])
        self.assertEqual(data["id"], "error")
        self.assertEqual(data["error"], "content too long (10000 chars, max 8000)")

    @patch("mnemon._run_mnemon")
    def test_remember_clamping(self, mock_run):
        mock_run.return_value = (0, MNEMON_OK, "")
        idx_path = Path("/tmp/test_mnemon_index_%d.json" % id(self))
        with patch.object(self.p, "_index_path", idx_path):
            if idx_path.exists():
                try:
                    idx_path.unlink()
                except Exception:
                    pass
            self.p.handle_tool_call("mnemon_remember", {"text": "foo", "importance": 8})
        called_args = mock_run.call_args[0][0]
        self.assertIn("--imp", called_args)
        self.assertEqual(called_args[called_args.index("--imp") + 1], "5")
        self.assertEqual(mock_run.call_args[1].get("store"), "test-store")

    @patch("mnemon._run_mnemon")
    def test_recall_returns_hits(self, mock_run):
        mock_run.return_value = (0, RECALL_OK, "")
        result = self.p.handle_tool_call("mnemon_recall", {"query": "memory", "limit": 5})
        data = json.loads(result)
        self.assertEqual(data["count"], 2)
        self.assertEqual(len(data["hits"]), 2)
        self.assertGreaterEqual(data["hits"][0]["score"], 0)
        self.assertEqual(mock_run.call_args[1].get("store"), "test-store")

    def test_recall_empty_query(self):
        result = self.p.handle_tool_call("mnemon_recall", {"query": ""})
        data = json.loads(result)
        self.assertIn("error", data)

    @patch("mnemon._run_mnemon")
    def test_forget_success(self, mock_run):
        mock_run.return_value = (0, FORGET_OK, "")
        idx_path = Path("/tmp/test_mnemon_index_%d.json" % id(self))
        with patch.object(self.p, "_index_path", idx_path):
            # write a dummy index
            idx_path.write_text('{"ids":{"i1":{"ts":"now","text":"x"}}}')
            result = self.p.handle_tool_call("mnemon_forget", {"insight_id": "i1"})
        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertIn("i1", data["forgotten"])
        self.assertEqual(mock_run.call_args[1].get("store"), "test-store")

    def test_forget_missing_id(self):
        result = self.p.handle_tool_call("mnemon_forget", {"insight_id": ""})
        data = json.loads(result)
        self.assertIn("error", data)

    @patch("mnemon._run_mnemon")
    def test_forget_cli_error(self, mock_run):
        mock_run.return_value = (1, "", "not found")
        result = self.p.handle_tool_call("mnemon_forget", {"insight_id": "bad-id"})
        data = json.loads(result)
        self.assertIn("error", data)


# ---------------------------------------------------------------------------
# Tests: sync_turn / auto-categorisation
# ---------------------------------------------------------------------------

class TestAutoCategorise(unittest.TestCase):

    def test_decision_keywords(self):
        p = MnemonMemoryProvider()
        for kw in ["choose", "decided", "picked", "selected", "chose"]:
            cat = p._auto_cat(f"I {kw} Python over JS")
            self.assertEqual(cat, "decision")

    def test_general_default(self):
        p = MnemonMemoryProvider()
        self.assertEqual(p._auto_cat("The sky is blue"), "general")

    @patch("mnemon._run_mnemon")
    def test_forget_keyword_skips_auto_remember(self, mock_run):
        """Text containing 'forget' must NOT be auto-remembered."""
        p = MnemonMemoryProvider()
        with patch.object(p, "_remember_and_index") as mock_idx:
            p._auto_remember("forget about that", "ok")
        mock_idx.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: intent map
# ---------------------------------------------------------------------------

class TestIntentMap(unittest.TestCase):

    def test_why_detection(self):
        p = MnemonMemoryProvider()
        self.assertEqual(p._detect_intent("why is the sky blue"), "WHY")

    def test_when_detection(self):
        p = MnemonMemoryProvider()
        self.assertEqual(p._detect_intent("when does it end"), "WHEN")

    def test_entity_detection(self):
        p = MnemonMemoryProvider()
        self.assertEqual(p._detect_intent("who wrote this?"), "ENTITY")

    def test_general_default(self):
        p = MnemonMemoryProvider()
        self.assertEqual(p._detect_intent("tell me about recursion"), "GENERAL")


# ---------------------------------------------------------------------------
# Tests: Config and Setup
# ---------------------------------------------------------------------------

class TestConfigAndSetup(unittest.TestCase):

    def test_get_config_schema(self):
        p = MnemonMemoryProvider()
        schema = p.get_config_schema()
        self.assertIsInstance(schema, list)
        self.assertEqual(len(schema), 3)
        self.assertEqual(schema[0]["key"], "store")
        self.assertEqual(schema[1]["key"], "max_compress_chars")
        self.assertEqual(schema[2]["key"], "max_mirror_chars")

    @patch("mnemon._run_mnemon")
    def test_save_config_and_initialize_with_hermes_home(self, mock_run):
        import shutil
        import tempfile
        mock_run.return_value = (0, "", "")
        
        # Create a temporary directory for hermes_home
        tmpdir = tempfile.mkdtemp()
        try:
            p = MnemonMemoryProvider()
            
            # 1. Save config
            p.save_config({"store": "my-custom-store"}, tmpdir)
            config_file = Path(tmpdir) / "mnemon.json"
            self.assertTrue(config_file.exists())
            
            # 2. Initialize should load the config
            p.initialize("ses-xyz", agent_identity="default-profile", hermes_home=tmpdir)
            self.assertEqual(p._store, "my-custom-store")
            self.assertEqual(p._index_path, Path(tmpdir) / "mnemon_id_index.json")
            
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Tests: CLI Extension
# ---------------------------------------------------------------------------

class TestCliExtension(unittest.TestCase):

    @patch("mnemon.cli._run_mnemon")
    @patch("sys.stdout", new_callable=__import__("io").StringIO)
    def test_cli_status_success(self, mock_stdout, mock_run):
        from mnemon.cli import handle_mnemon_command
        mock_run.side_effect = [
            (0, "mnemon v0.1.3", ""),       # --version
            (0, "default\ntest-store", ""), # store list
        ]
        args = MagicMock()
        args.mnemon_cmd = "status"
        parser = MagicMock()
        
        with patch.dict("os.environ", {"MNEMON_STORE": "test-store"}, clear=False):
            with self.assertRaises(SystemExit) as cm:
                handle_mnemon_command(args, parser)
            self.assertEqual(cm.exception.code, 0)
            
        output = mock_stdout.getvalue()
        self.assertIn("Status: ACTIVE", output)
        self.assertIn("Version: mnemon v0.1.3", output)
        self.assertIn("Active Store: test-store", output)

    @patch("mnemon.cli._run_mnemon")
    @patch("sys.stdout", new_callable=__import__("io").StringIO)
    def test_cli_config_success(self, mock_stdout, mock_run):
        import shutil
        import tempfile

        from mnemon.cli import handle_mnemon_command
        
        args = MagicMock()
        args.mnemon_cmd = "config"
        parser = MagicMock()
        
        tmpdir = tempfile.mkdtemp()
        config_file = Path(tmpdir) / ".hermes" / "mnemon.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text('{"store": "configured-store"}')
        
        try:
            with patch("mnemon.cli.Path.home", return_value=Path(tmpdir)):
                with self.assertRaises(SystemExit) as cm:
                    handle_mnemon_command(args, parser)
                self.assertEqual(cm.exception.code, 0)
            
            output = mock_stdout.getvalue()
            self.assertIn("configured-store", output)
        finally:
            shutil.rmtree(tmpdir)

    @patch("mnemon.cli._run_mnemon")
    @patch("sys.stdout", new_callable=__import__("io").StringIO)
    def test_cli_forget_success(self, mock_stdout, mock_run):
        import shutil
        import tempfile

        from mnemon.cli import handle_mnemon_command
        
        mock_run.return_value = (0, "forgotten", "")
        args = MagicMock()
        args.mnemon_cmd = "forget"
        args.insight_id = "uuid-123"
        parser = MagicMock()
        
        tmpdir = tempfile.mkdtemp()
        index_file = Path(tmpdir) / ".hermes" / "mnemon_id_index.json"
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index_file.write_text('{"ids": {"uuid-123": {"ts": "now"}}}')
        
        try:
            with patch("mnemon.cli.Path.home", return_value=Path(tmpdir)):
                with self.assertRaises(SystemExit) as cm:
                    handle_mnemon_command(args, parser)
                self.assertEqual(cm.exception.code, 0)
                
            output = mock_stdout.getvalue()
            self.assertIn("Successfully requested soft-delete", output)
            self.assertIn("Removed from local ID index", output)
            # Verify it's actually removed from index_file
            content = json.loads(index_file.read_text())
            self.assertNotIn("uuid-123", content["ids"])
        finally:
            shutil.rmtree(tmpdir)

    def test_register_cli(self):
        from mnemon.cli import register_cli
        subparser = MagicMock()
        register_cli(subparser)
        subparser.add_parser.assert_called_once_with("mnemon", help="Mnemon memory provider commands")


# ---------------------------------------------------------------------------
# Issue #17 contextvar contract: background jobs MUST go through
# agent.memory_provider.spawn_context_thread (see the in-tree reference
# provider plugins/memory/supermemory/__init__.py lines 15 and 494), not
# bare threading.Thread. A bare thread starts with empty contextvars and,
# under multiplexed profiles, would silently land on the default profile's
# store (or fail closed on get_secret()).
# ---------------------------------------------------------------------------

class TestSpawnContextThread(unittest.TestCase):

    def setUp(self):
        self.p = MnemonMemoryProvider()
        self.p._store = "test-store"
        self.p._session_id = "s1"
        # initialize() normally sets these from hermes_home config; emulate here
        self.p._max_compress_chars = 600
        self.p._max_mirror_chars = 8000

    @patch("mnemon.spawn_context_thread")
    def test_queue_prefetch_uses_spawn_context_thread(self, mock_spawn):
        self.p.queue_prefetch("query", session_id="s1")
        mock_spawn.assert_called_once_with(
            self.p.prefetch,
            name="mnemon-prefetch",
            args=("query",),
            kwargs={"session_id": "s1"},
        )

    @patch("mnemon.spawn_context_thread")
    def test_sync_turn_uses_spawn_context_thread(self, mock_spawn):
        self.p.sync_turn("user said x", "agent replied y", session_id="s1")
        mock_spawn.assert_called_once_with(
            self.p._auto_remember,
            name="mnemon-sync",
            args=("user said x", "agent replied y"),
        )

    @patch("mnemon.spawn_context_thread")
    def test_on_memory_write_uses_spawn_context_thread(self, mock_spawn):
        with patch.object(self.p, "_remember_and_index") as mock_idx:
            mock_idx.return_value = "uuid-1"
            self.p.on_memory_write("add", "user", "remember this")
            # the target is a closure; exec it inside the patch so it hits the mock
            call = mock_spawn.call_args
            target = call.args[0] if call.args else call.kwargs.get("target")
            self.assertTrue(callable(target))
            target()  # should not raise
        mock_spawn.assert_called_once()
        self.assertEqual(call.kwargs.get("name"), "mnemon-mirror")
        mock_idx.assert_called_once()
        _, kw = mock_idx.call_args
        self.assertEqual(kw.get("category"), "preference")
        self.assertEqual(kw.get("importance"), 4)
        self.assertEqual(kw.get("source"), "agent")


if __name__ == "__main__":
    unittest.main()
