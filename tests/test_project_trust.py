"""
Tests for project trust (seacode.project_trust), the .git/ write guard
and the `git config` guard.

A cloned repository can name programs to launch (.mcp.json servers,
seacode.toml hooks); SeaCode asks before running them and remembers the
answer for that exact config only. The model may not write into .git/
(hooks and config there make git run programs) or change git config
through run_command.

Run with:  python3 -m pytest tests/test_project_trust.py -v
       or: python3 -m unittest tests.test_project_trust -v
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.cli as cli_module
import seacode.hooks as hooks_module
from seacode import project_trust as trust
from seacode import tools as tools_module


MCP_CONFIG = {
    "files": {
        "command": "python",
        "args": ["server.py"],
        "env": {"API_TOKEN": "super-secret-value"},
    }
}


class TrustTestCase(unittest.TestCase):
    """Temp project + temp per-user trust store."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        base = Path(self._tmp.name).resolve()
        self.root = base / "project"
        self.root.mkdir()
        self.store = base / "config" / "trusted.json"

        env = patch.dict(os.environ, {
            "SEACODE_TRUST_FILE": str(self.store),
            "SEACODE_DIR": str(self.root),
        })
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SEACODE_TRUST_PROJECT", None)

        self.addCleanup(hooks_module.clear_hooks)

    def write_mcp(self, servers=None):
        (self.root / ".mcp.json").write_text(
            json.dumps({"mcpServers": servers or MCP_CONFIG})
        )

    def confirm(self, payload, answer="", **kwargs):
        asked = []

        def ask(prompt):
            asked.append(prompt)
            return answer

        shown = []
        result = trust.confirm(
            self.root, trust.KIND_MCP, payload, ["files: python server.py"],
            interactive=kwargs.pop("interactive", True),
            ask=ask, out=shown.append, **kwargs,
        )

        return result, asked, shown


class FingerprintTests(unittest.TestCase):

    def test_key_order_does_not_matter(self):
        self.assertEqual(
            trust.fingerprint({"a": 1, "b": [1, 2]}),
            trust.fingerprint({"b": [1, 2], "a": 1}),
        )

    def test_any_change_changes_it(self):
        base = trust.fingerprint({"s": {"command": "python", "args": ["a"]}})

        self.assertNotEqual(
            base,
            trust.fingerprint({"s": {"command": "python", "args": ["b"]}}),
        )


class ConfirmTests(TrustTestCase):

    def test_nothing_to_run_needs_no_prompt(self):
        result, asked, _ = self.confirm({}, interactive=False)

        self.assertTrue(result)
        self.assertEqual(asked, [])

    def test_untrusted_and_non_interactive_fails_closed(self):
        err = io.StringIO()

        with redirect_stderr(err):
            result, asked, _ = self.confirm(
                {"files": {}}, interactive=False
            )

        self.assertFalse(result)
        self.assertEqual(asked, [])
        self.assertIn("skipping untrusted", err.getvalue())

    def test_yes_runs_once_without_remembering(self):
        payload = {"files": {"command": "python"}}

        result, asked, _ = self.confirm(payload, answer="y")

        self.assertTrue(result)
        self.assertEqual(len(asked), 1)
        self.assertFalse(trust.is_trusted(self.root, "mcp", payload))
        self.assertFalse(self.store.exists())

        again, asked_again, _ = self.confirm(payload, answer="n")
        self.assertFalse(again)
        self.assertEqual(len(asked_again), 1)

    def test_always_remembers_and_stops_asking(self):
        payload = {"files": {"command": "python"}}

        result, _, _ = self.confirm(payload, answer="a")
        self.assertTrue(result)

        again, asked, _ = self.confirm(payload, answer="n")
        self.assertTrue(again)
        self.assertEqual(asked, [])

    def test_changed_config_asks_again(self):
        self.confirm({"files": {"command": "python"}}, answer="a")

        result, asked, _ = self.confirm(
            {"files": {"command": "python", "args": ["evil.py"]}},
            answer="n",
        )

        self.assertFalse(result)
        self.assertEqual(len(asked), 1)

    def test_default_answer_is_no(self):
        for answer in ("", "n", "no", "maybe"):
            result, _, _ = self.confirm({"x": {}}, answer=answer)
            self.assertFalse(result, answer)

    def test_eof_is_no(self):
        def ask(prompt):
            raise EOFError

        self.assertFalse(trust.confirm(
            self.root, "mcp", {"x": {}}, ["x"],
            interactive=True, ask=ask, out=lambda *_: None,
        ))

    def test_explicit_opt_in_skips_prompt_but_does_not_remember(self):
        payload = {"x": {}}

        result, asked, _ = self.confirm(
            payload, interactive=False, opt_in=True
        )

        self.assertTrue(result)
        self.assertEqual(asked, [])
        self.assertFalse(trust.is_trusted(self.root, "mcp", payload))

    def test_env_opt_in(self):
        os.environ["SEACODE_TRUST_PROJECT"] = "1"

        result, asked, _ = self.confirm({"x": {}}, interactive=False)

        self.assertTrue(result)
        self.assertEqual(asked, [])

    def test_prompt_lists_commands(self):
        _, _, shown = self.confirm({"x": {}}, answer="n")

        self.assertTrue(any("files: python server.py" in l for l in shown))

    def test_approval_is_per_project(self):
        payload = {"x": {}}
        self.confirm(payload, answer="a")

        other = self.root.parent / "other"
        other.mkdir()

        self.assertFalse(trust.is_trusted(other, "mcp", payload))

    def test_mcp_and_hooks_approved_independently(self):
        payload = {"x": {}}
        self.confirm(payload, answer="a")

        self.assertFalse(trust.is_trusted(self.root, "hooks", payload))

    def test_forget(self):
        payload = {"x": {}}
        self.confirm(payload, answer="a")

        self.assertTrue(trust.forget(self.root))
        self.assertFalse(trust.is_trusted(self.root, "mcp", payload))
        self.assertFalse(trust.forget(self.root))

    def test_corrupt_store_is_untrusted_not_a_crash(self):
        self.store.parent.mkdir(parents=True)
        self.store.write_text("{not json")

        self.assertFalse(trust.is_trusted(self.root, "mcp", {"x": {}}))

    def test_store_lives_outside_the_project(self):
        self.confirm({"x": {}}, answer="a")

        self.assertTrue(self.store.exists())
        self.assertFalse(
            self.store.resolve().is_relative_to(self.root)
        )

    def test_project_cannot_ship_its_own_approval(self):
        payload = MCP_CONFIG
        (self.root / "trusted_projects.json").write_text(json.dumps({
            "projects": {
                str(self.root): {"mcp": trust.fingerprint(payload)}
            }
        }))

        self.assertFalse(trust.is_trusted(self.root, "mcp", payload))


class DescribeTests(unittest.TestCase):

    def test_env_values_are_never_shown(self):
        text = "\n".join(trust.describe_mcp({
            "files": {
                "command": "python",
                "args": ["server.py", "--flag"],
                "env": {"API_TOKEN": "super-secret-value"},
            }
        }))

        self.assertIn("python server.py --flag", text)
        self.assertIn("API_TOKEN", text)
        self.assertNotIn("super-secret-value", text)

    def test_describe_hooks(self):
        lines = trust.describe_hooks({
            "pre_tool": ["python guard.py"], "post_tool": ["python log.py"],
        })

        self.assertEqual(
            lines, ["pre_tool: python guard.py", "post_tool: python log.py"]
        )


class CliGateTests(TrustTestCase):
    """The wiring in seacode.cli: untrusted config is never started."""

    def setUp(self):
        super().setUp()
        self.addCleanup(cli_module._stop_mcp)

    def test_untrusted_mcp_is_not_spawned(self):
        self.write_mcp()

        with patch.object(
            cli_module.mcp_module, "MCPManager"
        ) as manager, patch("sys.stdin.isatty", return_value=False), \
                redirect_stderr(io.StringIO()):
            cli_module._start_mcp(self.root)

        manager.assert_not_called()

    def test_trusted_mcp_is_spawned(self):
        self.write_mcp()
        trust.remember(
            self.root, trust.KIND_MCP,
            cli_module.mcp_module.load_mcp_config(self.root),
        )

        with patch.object(
            cli_module.mcp_module, "MCPManager"
        ) as manager:
            cli_module._start_mcp(self.root)

        manager.return_value.start_all.assert_called_once_with(self.root)

    def test_opt_in_spawns_without_prompt(self):
        self.write_mcp()

        with patch.object(
            cli_module.mcp_module, "MCPManager"
        ) as manager, patch("builtins.input") as prompt:
            cli_module._start_mcp(self.root, opt_in=True)

        prompt.assert_not_called()
        manager.return_value.start_all.assert_called_once()

    def test_no_mcp_json_spawns_nothing_and_asks_nothing(self):
        with patch("builtins.input") as prompt:
            self.assertTrue(cli_module._mcp_trusted(self.root))

        prompt.assert_not_called()

    def test_yes_flag_does_not_imply_project_trust(self):
        # --yes only sets the model's permission level.
        self.write_mcp()
        args = SimpleNamespace(yes=True, trust_project=False)

        self.assertFalse(getattr(args, "trust_project"))

        with patch("sys.stdin.isatty", return_value=False), \
                redirect_stderr(io.StringIO()):
            self.assertFalse(cli_module._mcp_trusted(
                self.root, getattr(args, "trust_project", False)
            ))

    def test_untrusted_hooks_are_not_installed(self):
        hooks = {"pre_tool": ["python evil.py"], "post_tool": []}

        with patch("sys.stdin.isatty", return_value=False), \
                redirect_stderr(io.StringIO()):
            self.assertFalse(cli_module._hooks_trusted(self.root, hooks))

    def test_trusted_hooks_are_installed(self):
        hooks = {"pre_tool": ["python guard.py"], "post_tool": []}
        trust.remember(self.root, trust.KIND_HOOKS, {
            k: v for k, v in hooks.items() if v
        })

        self.assertTrue(cli_module._hooks_trusted(self.root, hooks))

    def test_empty_hooks_need_no_trust(self):
        self.assertTrue(cli_module._hooks_trusted(
            self.root, {"pre_tool": [], "post_tool": []}
        ))

    def test_run_mcp_refuses_untrusted_project(self):
        self.write_mcp()
        args = SimpleNamespace(dir=None, trust_project=False)

        out = io.StringIO()

        with patch.object(
            cli_module.mcp_module, "MCPManager"
        ) as manager, patch("sys.stdin.isatty", return_value=False), \
                redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cli_module.run_mcp(args)

        self.assertEqual(code, 1)
        manager.assert_not_called()
        self.assertIn("not trusted", out.getvalue())


class TrustCommandTests(TrustTestCase):

    def run_trust(self, answer=None, revoke=False):
        args = SimpleNamespace(dir=None, revoke=revoke)
        out = io.StringIO()

        with patch("builtins.input", return_value=answer or ""), \
                redirect_stdout(out):
            code = cli_module.run_trust(args)

        return code, out.getvalue()

    def test_parser_knows_trust_and_flags(self):
        parser = cli_module.build_parser()

        self.assertTrue(parser.parse_args(["--trust-project"]).trust_project)
        self.assertTrue(
            parser.parse_args(["repl", "--trust-project"]).trust_project
        )
        self.assertFalse(parser.parse_args(["repl"]).trust_project)
        self.assertTrue(parser.parse_args(["trust", "--revoke"]).revoke)
        self.assertTrue(
            parser.parse_args(["mcp", "--trust-project"]).trust_project
        )

    def test_nothing_to_trust(self):
        code, text = self.run_trust()

        self.assertEqual(code, 0)
        self.assertIn("Nothing to trust", text)

    def test_approving_saves_the_exact_config(self):
        self.write_mcp()

        code, _ = self.run_trust("y")

        self.assertEqual(code, 0)
        self.assertTrue(trust.is_trusted(
            self.root, trust.KIND_MCP,
            cli_module.mcp_module.load_mcp_config(self.root),
        ))

    def test_declining_saves_nothing(self):
        self.write_mcp()

        code, _ = self.run_trust("n")

        self.assertEqual(code, 1)
        self.assertFalse(self.store.exists())

    def test_revoke(self):
        self.write_mcp()
        self.run_trust("y")

        code, text = self.run_trust(revoke=True)

        self.assertEqual(code, 0)
        self.assertIn("Forgot", text)
        self.assertFalse(trust.is_trusted(
            self.root, trust.KIND_MCP,
            cli_module.mcp_module.load_mcp_config(self.root),
        ))

    def test_hooks_covered_too(self):
        (self.root / "seacode.toml").write_text(
            '[hooks]\npre_tool = ["python guard.py"]\n'
        )

        code, _ = self.run_trust("y")

        self.assertEqual(code, 0)
        self.assertTrue(cli_module._hooks_trusted(
            self.root, hooks_module.load_hooks(self.root)
        ))


# -- .git/ write guard ------------------------------------------------


class GitGuardTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        self.root = Path(self._tmp.name).resolve()
        (self.root / ".git" / "hooks").mkdir(parents=True)
        (self.root / ".git" / "config").write_text("[core]\n")
        (self.root / "a.txt").write_text("hello\n")

        env = patch.dict(os.environ, {"SEACODE_DIR": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

        # Approve everything, to prove the guard is not the prompt.
        perm = patch.object(
            tools_module, "request_permission", return_value=True
        )
        perm.start()
        self.addCleanup(perm.stop)


class GitPathTests(GitGuardTestCase):

    def test_is_git_path(self):
        for rel in (
            ".git/hooks/pre-commit",
            ".git/config",
            ".git",
            "sub/.git/config",
            ".GIT/config",
        ):
            self.assertTrue(
                tools_module.is_git_path(self.root / rel, self.root), rel
            )

    def test_lookalikes_are_not_git_paths(self):
        for rel in (
            ".gitignore", ".github/workflows/ci.yml", "a.git",
            "git/config", ".gitattributes", "x/legit.git/y",
        ):
            self.assertFalse(
                tools_module.is_git_path(self.root / rel, self.root), rel
            )


class GitWriteBlockedTests(GitGuardTestCase):

    def test_write_file_cannot_create_a_hook(self):
        with self.assertRaises(PermissionError):
            tools_module.write_file(
                ".git/hooks/pre-commit", "#!/bin/sh\necho pwned\n"
            )

        self.assertFalse(
            (self.root / ".git" / "hooks" / "pre-commit").exists()
        )

    def test_write_file_cannot_edit_git_config(self):
        with self.assertRaises(PermissionError):
            tools_module.write_file(
                ".git/config", "[core]\nfsmonitor = evil\n"
            )

        self.assertEqual(
            (self.root / ".git" / "config").read_text(), "[core]\n"
        )

    def test_write_file_blocks_other_spellings(self):
        for rel in (".GIT/hooks/x", "./.git/hooks/x", "sub/../.git/hooks/x"):
            with self.assertRaises(PermissionError, msg=rel):
                tools_module.write_file(rel, "x")

    def test_write_file_via_symlink_into_git_is_blocked(self):
        link = self.root / "hooks_link"

        try:
            link.symlink_to(self.root / ".git" / "hooks")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")

        with self.assertRaises(PermissionError):
            tools_module.write_file("hooks_link/pre-commit", "x")

    def test_replace_in_file_blocked(self):
        with self.assertRaises(PermissionError):
            tools_module.replace_in_file(".git/config", "[core]", "[evil]")

        self.assertEqual(
            (self.root / ".git" / "config").read_text(), "[core]\n"
        )

    def test_delete_file_blocked(self):
        with self.assertRaises(PermissionError):
            tools_module.delete_file(".git/config")

        self.assertTrue((self.root / ".git" / "config").exists())

    def test_internal_helpers_blocked(self):
        with self.assertRaises(PermissionError):
            tools_module.restore_file_content(".git/config", "x")

        with self.assertRaises(PermissionError):
            tools_module.remove_created_file(".git/config")

        self.assertTrue((self.root / ".git" / "config").exists())

    def test_undo_will_not_restore_into_git(self):
        # A stale backup whose target is under .git/ must not be applied.
        backup_dir = (
            self.root / tools_module.BACKUP_DIR_NAME / ".git" / "config"
        )
        backup_dir.mkdir(parents=True)
        (backup_dir / "20260101-000000-000000.bak").write_text("[evil]\n")

        with self.assertRaises(PermissionError):
            tools_module.undo_last_change(".git/config")

        self.assertEqual(
            (self.root / ".git" / "config").read_text(), "[core]\n"
        )

    def test_normal_files_still_work(self):
        tools_module.write_file(".gitignore", ".venv/\n")
        tools_module.write_file(".github/ci.yml", "on: push\n")
        tools_module.replace_in_file("a.txt", "hello", "bye")

        self.assertEqual((self.root / ".gitignore").read_text(), ".venv/\n")
        self.assertEqual((self.root / "a.txt").read_text(), "bye\n")


# -- git config guard ---------------------------------------------------


class GitConfigGuardTests(GitGuardTestCase):

    def check(self, *args):
        tools_module.validate_command_args(["git", *args], self.root)

    def assertBlocked(self, *args):
        with self.assertRaises(PermissionError):
            self.check(*args)

    def test_config_writes_blocked(self):
        self.assertBlocked("config", "core.hooksPath", "/tmp/hooks")
        self.assertBlocked("config", "core.fsmonitor", "evil")
        self.assertBlocked("config", "--add", "alias.x", "!sh")
        self.assertBlocked("config", "--unset", "user.name")
        self.assertBlocked("config", "--global", "core.editor", "evil")

    def test_config_after_global_option_blocked(self):
        self.assertBlocked("-C", ".", "config", "core.hooksPath", "x")
        self.assertBlocked("--no-pager", "config", "core.pager", "evil")

    def test_config_reads_allowed(self):
        self.check("config", "--get", "user.name")
        self.check("config", "--list")
        self.check("config", "-l")
        self.check("config", "--get-regexp", "alias")

    def test_other_subcommands_unaffected(self):
        self.check("status")
        self.check("log", "-c")
        self.check("diff", "--stat")
        self.check("commit", "-m", "config")  # "config" as a value
        self.check("log", "--grep", "config")

    def test_run_command_refuses_before_prompting(self):
        with patch.object(
            tools_module, "request_permission"
        ) as prompt, self.assertRaises(PermissionError):
            tools_module.run_command("git config core.hooksPath /x")

        prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
