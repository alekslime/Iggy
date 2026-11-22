"""Post- and pre-execution verification for agent modifications (the
M7 guardrail series).

Extracted from agent.py: the agent loop owns *recovery policy* (retry
budgets, corrective messages, transcript handling); this module owns
the pure checks themselves. Every check is a cheap mechanical test --
regex/list/AST based, no model calls -- that returns None when the
modification looks fine, or a human-readable explanation when it
doesn't. Keeping them here makes each failure class unit-testable in
isolation and keeps the loop readable.

Failure classes:
  M7.6  semantic edit verification (destructive / wrong content)
  M7.7  wrong-file edit detection
  M7.8  false verification-claim detection
  M7.9  unread-file edit guard (pre-execution)
  M7.10 Python syntax validation
  M7.11 incomplete multi-target edit detection
  M7.12 scope-creep detection
"""

import ast
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

import seacode.tools as _tools
from seacode.tools import list_files, read_file


# Tools whose failure represents an incomplete modification, not just
# an information-gathering step. Used to detect and recover from
# failed edits (M7.3).
MODIFICATION_TOOLS = {"write_file", "replace_in_file", "delete_file"}

# Phrases that indicate the user explicitly wants the change confined
# to specific, named file(s) -- e.g. "only change `app/config.py`,
# don't touch anything else." M7.12's scope-creep check only fires when
# one of these is present: touching files beyond what was named is
# completely normal for many legitimate requests (a bug fix pulling in
# a related import, an accompanying test update, etc), so without an
# explicit signal from the user this would false-positive constantly.
# Requiring an explicit keyword trades recall for precision on
# purpose -- this is expected to miss plenty of real scope creep that
# isn't preceded by one of these phrases.
SCOPE_LIMIT_KEYWORDS = (
    "only change",
    "only modify",
    "only edit",
    "only touch",
    "only update",
    "just change",
    "just modify",
    "just edit",
    "just touch",
    "only that file",
    "only this file",
    "only in that file",
    "only in this file",
    "nothing else",
    "don't touch anything else",
    "do not touch anything else",
    "don't modify anything else",
    "do not modify anything else",
    "don't touch any other file",
    "do not touch any other file",
    "no other files",
    "minimal change",
    "smallest possible change",
)

# Tool calls that plausibly check a modification's result: actually
# running something, or re-reading/searching the file to look at it
# again. list_files and git_status don't count -- they don't inspect
# the change itself.
VERIFICATION_TOOLS = frozenset({"run_command", "read_file", "search_files"})

# Phrases that indicate the final answer is asserting verification
# happened. Practical keyword list, not NLP -- like M7.6's rewrite-
# intent keywords, this accepts known misses (e.g. a negated claim
# like "I have not verified this") as a documented limitation rather
# than trying to parse negation.
VERIFICATION_CLAIM_KEYWORDS = (
    "verified",
    "verification passed",
    "confirmed it works",
    "confirmed that it works",
    "confirmed the fix",
    "tests pass",
    "test passes",
    "tests passed",
    "ran the tests",
    "ran the test suite",
    "successfully tested",
    "tested and it works",
    "tested and confirmed",
    "i checked and it works",
)

# Fraction of a file's original content that must still be recognizable
# afterwards for a modification to NOT be considered destructive, used
# by content_preserved_ratio(). Two thresholds: edits with a clearly
# identified small target ("from X to Y") are held to a much stricter
# bar than edits where we couldn't extract an explicit target, since we
# have stronger evidence the change was supposed to be tiny.
TARGETED_EDIT_PRESERVATION_THRESHOLD = 0.7
GENERIC_PRESERVATION_THRESHOLD = 0.5

# Below this many characters, percentage-of-content-preserved is too
# noisy to be a meaningful destructive-change signal on its own (a
# single-line file can "lose" 50%+ from a completely ordinary edit).
# Only applies when we have no explicit edit target to anchor on --
# targeted edits (an identified old/new pair) are checked regardless
# of file size, since we have a much more specific expectation there.
MIN_LENGTH_FOR_GENERIC_DESTRUCTIVE_CHECK = 200

# Phrases that indicate the user actually wants a full-file rewrite, so
# M7.6's destructive-change guard/verification should not fire.
REWRITE_INTENT_KEYWORDS = (
    "rewrite the whole file",
    "rewrite the entire file",
    "rewrite entire file",
    "replace the whole file",
    "replace the entire file",
    "entire file",
    "whole file",
    "from scratch",
    "start over",
    "regenerate the file",
    "recreate the file",
    "completely rewrite",
    "full rewrite",
    "rewrite it completely",
)

# M7.6 uses a small library of regex patterns (not a natural-language
# parser) to recognize a handful of common ways people phrase a small,
# targeted textual edit. Each pattern's capture groups are labeled with
# which role ("old" and/or "new") they play; some phrasings only name
# the desired result text and don't spell out what's being replaced.
_QUOTE = r"[`'\"]([^`'\"]+)[`'\"]"

CHANGE_PAIR_PATTERNS = (
    # "... from `X` to `Y`" / "... change `X` to `Y`"
    (
        re.compile(
            rf"(?:from|change)\s+{_QUOTE}\s+to\s+{_QUOTE}",
            re.IGNORECASE | re.DOTALL,
        ),
        ("old", "new"),
    ),
    # "replace `X` with `Y`"
    (
        re.compile(
            rf"replace\s+{_QUOTE}\s+with\s+{_QUOTE}",
            re.IGNORECASE | re.DOTALL,
        ),
        ("old", "new"),
    ),
    # "`Y` instead of `X`"
    (
        re.compile(
            rf"{_QUOTE}\s+instead\s+of\s+{_QUOTE}",
            re.IGNORECASE | re.DOTALL,
        ),
        ("new", "old"),
    ),
    # "... to/should say `Y`" -- names the desired result, not the
    # original text.
    (
        re.compile(
            rf"(?:to|should)\s+say\s+{_QUOTE}",
            re.IGNORECASE | re.DOTALL,
        ),
        ("new",),
    ),
    # "say `Y` instead" -- same idea, opposite word order.
    (
        re.compile(
            rf"say\s+{_QUOTE}\s+instead\b",
            re.IGNORECASE | re.DOTALL,
        ),
        ("new",),
    ),
)

# M7.7 uses a small allowlist of common source/config extensions to
# find filename-like tokens in a user's request (e.g. "utils.py",
# "seacode/tools.py"). Restricting to known extensions -- rather than
# matching any "word.word" pattern -- avoids false positives like
# "e.g." or a version number such as "v2.0".
_FILENAME_EXTENSIONS = (
    "py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|cfg|ini|sh|bash|"
    "css|s?css|html?|java|kt|swift|cpp|cc|cxx|c|h|hpp|go|rs|rb|"
    "php|sql|xml|csv|env|gitignore|dockerfile"
)

FILENAME_PATTERN = re.compile(
    rf"\b[\w\-./\\]*\.(?:{_FILENAME_EXTENSIONS})\b",
    re.IGNORECASE,
)

# Action verbs for M7.15's no-work-final-answer check: a request
# containing one of these is asking for something to be *done*, so
# a final answer before any tool has run is premature.
ACTION_REQUEST_VERBS = (
    "fix", "change", "update", "edit", "modify", "create", "delete",
    "remove", "add", "implement", "refactor", "rename", "move",
    "write", "run", "check", "test", "inspect", "look", "find",
    "search", "read", "show", "list", "explain",
)


def classify_modification_result(tool_name: str, result) -> str:
    """Classify the outcome of a modification tool call.

    Returns one of:
    - "success": the modification was actually applied.
    - "denied": the user declined permission (not a failure to retry).
    - "failed": a genuine, recoverable tool failure occurred.
    - "not_applicable": this tool call isn't a modification, or its
      result doesn't match a known pattern.

    This only inspects the exact strings write_file/replace_in_file/
    delete_file are documented to return, plus the {"error": ...}
    shape produced when execute_tool raises. It does not guess at
    other cases.
    """

    if tool_name not in MODIFICATION_TOOLS:
        return "not_applicable"

    if isinstance(result, dict) and "error" in result:
        return "failed"

    if not isinstance(result, str):
        return "not_applicable"

    if result == "File modification denied by user.":
        return "denied"

    if result.startswith("Replacement applied successfully:"):
        return "success"

    if result.startswith("File written successfully:"):
        return "success"

    if result.startswith("File deleted successfully:"):
        return "success"

    if result.startswith("Replacement failed:"):
        return "failed"

    if result.startswith("Delete failed:"):
        return "failed"

    return "not_applicable"


# -----------------------------------------------------------------------
# M7.6 -- semantic edit verification
# -----------------------------------------------------------------------
#
# The loop-level mechanisms (M7.3/M7.5) only know whether a tool call
# itself succeeded or failed. They can't tell that a `write_file` call
# which reported success actually destroyed the file it was supposed to
# make one small edit to. The helpers below add two cheap, non-AST
# checks:
#
#   1. extract_change_pair() / mentions_full_rewrite_intent() -- figure
#      out, from the user's own request, whether this looks like a
#      small targeted edit or an intentional full rewrite.
#   2. content_preserved_ratio() -- compare a file's content before and
#      after a modification to see how much of it survived.
#
# These feed into check_unnecessary_full_rewrite() (a pre-execution
# guard that steers `write_file` towards `replace_in_file`) and
# verify_semantic_edit() (a post-execution check run after any
# modification tool reports success).


def extract_change_pair(text: str):
    """Best-effort extraction of an edit target from a user request.

    Returns (old_text, new_text), where old_text may be None if the
    request identifies what the result should say without spelling out
    the exact text being replaced (e.g. "update the comment to say
    `Y`"). Returns None if nothing recognizable was found.

    Recognizes a handful of common phrasings via CHANGE_PAIR_PATTERNS.
    This is a practical heuristic for M7.6, not a natural-language
    parser -- it's expected to return None for most requests that
    don't spell out the target text in quotes.
    """

    if not isinstance(text, str):
        return None

    for pattern, roles in CHANGE_PAIR_PATTERNS:

        match = pattern.search(text)

        if not match:
            continue

        values = {}

        for role, group in zip(roles, match.groups()):
            values[role] = group.strip() if group else None

        new_text = values.get("new")
        old_text = values.get("old")

        if not new_text:
            continue

        if old_text is not None and (not old_text or old_text == new_text):
            continue

        return old_text, new_text

    return None


def mentions_full_rewrite_intent(text: str) -> bool:
    """Check whether a user request explicitly signals that a full-file
    rewrite is wanted, so destructive-change checks should stand down.
    """

    if not isinstance(text, str):
        return False

    lowered = text.lower()

    return any(keyword in lowered for keyword in REWRITE_INTENT_KEYWORDS)


def content_preserved_ratio(before: str, after: str) -> float:
    """Fraction of `before`'s content that is still recognizable inside
    `after`, based on difflib's matching-block sizes.

    1.0 means `after` contains everything `before` had (plus possibly
    more); 0.0 means essentially nothing in common. Used to catch a
    tiny requested edit that somehow wipes out most of the file.
    """

    if not before:
        return 1.0

    matcher = difflib.SequenceMatcher(None, before, after)
    matched = sum(block.size for block in matcher.get_matching_blocks())

    return matched / len(before)


def safe_read_file(path):
    """Read a project file, returning None instead of raising if it
    doesn't exist / isn't readable. Used to capture before/after content
    for M7.6 without disturbing normal tool error handling.
    """

    if not isinstance(path, str) or not path.strip():
        return None

    try:
        return read_file(path)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None


def check_unnecessary_full_rewrite(arguments: dict, last_user_request: str):
    """Guard run before executing `write_file`.

    If the file already exists, the user's request clearly names a
    small "from X to Y" edit, that exact old text occurs exactly once
    in the current file, and nothing suggests a full rewrite was
    actually wanted, this returns a corrective message instead of
    letting `write_file` proceed -- steering the model towards
    `replace_in_file` before anything touches disk.

    Returns None if the write_file call should proceed as-is (new file,
    no identifiable small-edit target, explicit rewrite intent, etc).
    """

    if not isinstance(arguments, dict):
        return None

    path = arguments.get("path")

    if not isinstance(path, str) or not path.strip():
        return None

    if mentions_full_rewrite_intent(last_user_request):
        return None

    pair = extract_change_pair(last_user_request)

    if pair is None:
        return None

    old_text, new_text = pair

    if old_text is None:
        # The request named the desired result but not the exact text
        # being replaced (e.g. "update the comment to say `Y`"), so
        # there's nothing concrete to check occurrence-count against.
        # Let write_file proceed; post-execution verification still
        # covers this case.
        return None

    current_content = safe_read_file(path)

    if current_content is None:
        # New file (or unreadable) -- write_file is the right tool.
        return None

    if current_content.count(old_text) != 1:
        # Not a clean single-occurrence match; replace_in_file would
        # refuse this too, so don't force the model into that dead end.
        return None

    return (
        f"This looks like a small, targeted textual change to an "
        f"existing file (`{path}`), not a full rewrite. The exact text "
        "being changed was found once in the current file.\n\n"
        "Use `replace_in_file` instead of `write_file` for this change:\n"
        f"  old_text: {json.dumps(old_text)}\n"
        f"  new_text: {json.dumps(new_text)}\n\n"
        "Do not rewrite the rest of the file's contents."
    )


def verify_semantic_edit(
    path: str,
    pre_content: str,
    post_content: str,
    last_user_request: str,
):
    """Post-execution check run after write_file/replace_in_file report
    success. Returns None if the result looks consistent with what was
    requested, or a human-readable explanation if it doesn't.

    Two independent checks, either of which can fail verification:
      - If the request named an explicit "from X to Y" edit, is the
        new text actually present in the result?
      - Regardless of that, did the edit destroy most of the file's
        prior content without the user asking for a full rewrite?
    """

    pair = extract_change_pair(last_user_request)

    if pair is not None:
        _, new_text = pair

        if new_text not in post_content:
            return (
                f"The requested replacement text was not found in "
                f"`{path}` after the modification.\n\n"
                f"Expected the file to contain:\n{json.dumps(new_text)}\n\n"
                "The modification tool reported success, but the "
                "requested change does not appear to have been applied "
                "correctly."
            )

    if not mentions_full_rewrite_intent(last_user_request):

        if pair is not None:
            threshold = TARGETED_EDIT_PRESERVATION_THRESHOLD
        elif len(pre_content) >= MIN_LENGTH_FOR_GENERIC_DESTRUCTIVE_CHECK:
            threshold = GENERIC_PRESERVATION_THRESHOLD
        else:
            threshold = None

        if threshold is not None:

            preserved = content_preserved_ratio(pre_content, post_content)

            if preserved < threshold:
                return (
                    f"This modification changed or removed roughly "
                    f"{round((1 - preserved) * 100)}% of `{path}`'s prior "
                    "content, which looks like an unrelated or destructive "
                    "rewrite rather than the targeted change that was "
                    "requested.\n\n"
                    "If a full rewrite was genuinely necessary, the request "
                    "should have said so explicitly. Otherwise, restore the "
                    "file's original structure and make only the specific "
                    "change requested, preferably with `replace_in_file`."
                )

    return None


# -----------------------------------------------------------------------
# M7.7 -- wrong-file edit detection
# -----------------------------------------------------------------------
#
# M7.6 checks whether an edit's *content* matches what was requested,
# but assumes the edit landed on the right file. It doesn't. The
# triggering failure class: the request names one file (explicitly or
# by a bare filename that's ambiguous across the project), but the
# model modifies a different one -- a similarly-named file elsewhere, a
# hallucinated path, or a stale target left over from earlier in the
# conversation. The edit itself can be flawless and still be wrong.
#
# Like M7.6, this is post-execution only and regex/list-based (no AST,
# no project-graph tracing of "which file the user meant"). It can only
# reason from filenames the user's own request actually mentions.


def extract_mentioned_filenames(text: str) -> set[str]:
    """Best-effort extraction of filename-like tokens from a user
    request (e.g. "fix the bug in utils.py" -> {"utils.py"}).

    Restricted to a common-extension allowlist (FILENAME_PATTERN) so
    this doesn't fire on unrelated "word.word" text. Returns an empty
    set if nothing recognizable was found -- callers should treat that
    as "cannot verify", not "no filename was intended".
    """

    if not isinstance(text, str):
        return set()

    return {
        match.strip("`'\" ").lstrip("./")
        for match in FILENAME_PATTERN.findall(text)
    }


def verify_target_file(path: str, last_user_request: str):
    """Post-execution check: does the file that was just modified match
    what the user's request actually named?

    Returns None if the result looks consistent with the request (or if
    there's not enough evidence in the request to check at all), or a
    human-readable explanation naming the correct file(s) if not.

    Only meaningful when the request mentions at least one filename --
    most requests don't spell one out explicitly ("fix the bug"), and
    for those this correctly returns None rather than guessing.
    """

    mentioned = extract_mentioned_filenames(last_user_request)

    if not mentioned:
        return None

    if not isinstance(path, str) or not path.strip():
        return None

    target_name = Path(path).name
    target_norm = path.replace("\\", "/")

    matches_target = any(
        Path(m).name == target_name or m.replace("\\", "/") == target_norm
        for m in mentioned
    )

    try:
        project_files = list_files(".")
    except Exception:
        # Can't enumerate the project -- nothing safe to check against.
        return None

    if matches_target:

        # Even a match can be ambiguous: the request may have named a
        # bare filename (no directory) that exists in more than one
        # place in the project, so hitting *a* file called that isn't
        # the same as hitting the *right* one.
        bare_mentions = {
            m for m in mentioned
            if Path(m).name == target_name and "/" not in m and "\\" not in m
        }

        if not bare_mentions:
            return None

        candidates = [f for f in project_files if Path(f).name == target_name]

        if len(candidates) <= 1:
            return None

        candidate_list = "\n".join(f"  - {c}" for c in candidates)

        return (
            f"The request mentioned `{target_name}`, but multiple files "
            f"with that name exist in the project:\n{candidate_list}\n\n"
            f"`{path}` was modified, but it isn't clear that's the one "
            "meant. Check for context clues (imports, surrounding code, "
            "directory structure) to determine the correct file, then "
            "redo the modification there. Revert this change if it "
            "turns out to be the wrong file."
        )

    # The edited file doesn't match anything the request mentioned --
    # look for a project file that does, to redirect to.
    redirect_candidates = []

    for mention in mentioned:
        mention_name = Path(mention).name

        for f in project_files:
            if Path(f).name == mention_name and f not in redirect_candidates:
                redirect_candidates.append(f)

    if not redirect_candidates:
        # Nothing in the project matches what was mentioned either --
        # not enough evidence to call this a wrong-file edit (e.g. a
        # new file being created, or a generic/example name).
        return None

    if len(redirect_candidates) == 1:
        return (
            f"The request mentioned `{redirect_candidates[0]}`, but the "
            f"modification was made to `{path}` instead, which doesn't "
            "match anything the request named.\n\n"
            f"Revert this change and apply it to "
            f"`{redirect_candidates[0]}` instead."
        )

    candidate_list = "\n".join(f"  - {c}" for c in redirect_candidates)

    return (
        f"The modification was made to `{path}`, which doesn't match "
        f"anything the request named. Project files matching what was "
        f"mentioned:\n{candidate_list}\n\n"
        "Check which one the request actually meant, then revert this "
        "change and redo the modification on the correct file."
    )


# -----------------------------------------------------------------------
# M7.10 -- syntax validation
# -----------------------------------------------------------------------
#
# M7.6/M7.7 both check whether an edit's *content* and *target* match
# what was requested, but neither one checks whether the result is
# actually valid code. A modification can pass both of those and still
# leave a .py file with a dangling paren, bad indentation, or an
# unclosed string -- a distinct failure class from "wrong content" or
# "wrong file".
#
# Scoped to .py files via ast.parse() -- stdlib, no subprocess, and
# unambiguous (it either parses or it doesn't), unlike the regex-based
# heuristics M7.6/M7.7 rely on. Other languages aren't covered: there's
# no free, dependency-free parser for them here, and guessing at syntax
# validity without a real parser would be worse than not checking at
# all.
#
# Deliberately checked against every .py file this run touches or
# creates, not just ones that parsed cleanly before the edit -- the
# bar is "nothing SeaCode leaves behind is broken Python," not "don't make
# an existing problem worse."


def verify_python_syntax(path: str, post_content: str):
    """Check whether a modified or newly created Python file parses as
    valid syntax.

    Returns None if the file isn't a .py file, or if it parses cleanly.
    Returns a human-readable message describing the SyntaxError
    otherwise.
    """

    if not isinstance(path, str) or not path.lower().endswith(".py"):
        return None

    if not isinstance(post_content, str):
        return None

    try:
        ast.parse(post_content, filename=path)
    except SyntaxError as e:
        return (
            f"`{path}` does not parse as valid Python after this "
            f"change: {e.msg} (line {e.lineno}, column {e.offset}).\n\n"
            "This would leave the project in a broken state."
        )

    return None


def verify_python_diagnostics(path: str):
    """Compile the on-disk file with the real interpreter
    (`python -m py_compile`) after an edit lands.

    ast.parse (verify_python_syntax) catches syntax errors in the
    content we captured; this catches what the actual toolchain
    says about the file as written -- the same signal a local LSP
    would give for this failure class, with no server needed.
    Returns None when the file compiles (or isn't Python, or the
    check itself can't run -- infra problems never block an edit).
    """

    if not isinstance(path, str) or not path.lower().endswith(".py"):
        return None

    try:
        root = _tools.get_project_root()
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(root / path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None

    if result.returncode == 0:
        return None

    detail = (result.stderr or result.stdout or "").strip()
    detail = detail[-600:] if detail else "py_compile failed"

    return (
        f"`{path}` does not compile with the project interpreter "
        f"after this change:\n{detail}\n\n"
        "This would leave the project in a broken state."
    )


# -----------------------------------------------------------------------
# M7.11 -- incomplete multi-target edit detection
# -----------------------------------------------------------------------
#
# M7.6/M7.7/M7.10 all check the quality of a *single* modification.
# None of them notice that a request asking for changes across several
# named files was only partially completed -- e.g. "rename `foo` to
# `bar` in `app/api.py` and `app/client.py`" but only `app/api.py`
# actually got touched.
#
# Deliberately narrow: this is a slice of "incomplete multi-step," not
# the whole failure class. It only fires when the request itself names
# two or more files (reusing M7.7's extract_mentioned_filenames) and at
# least one of them shows no successful, verified modification by the
# time a final answer is attempted. Multi-step tasks that aren't
# anchored to named files (e.g. "add a feature and write tests for
# it") aren't covered here -- there's no reliable, low-false-positive
# way to detect that with the same kind of mechanical check the rest
# of M7.x uses.


def check_incomplete_multi_target_edit(
    last_user_request: str,
    successfully_modified_paths: set,
):
    """Check whether a multi-file request still has an untouched file.

    Returns None if there's nothing to flag: fewer than two files
    named in the request, or no successful modification has happened
    at all yet this run (likely a pure information request, or an
    action request that hasn't started -- both already covered by
    other mechanisms). Otherwise returns a message listing the files
    that still need work.
    """

    mentioned = extract_mentioned_filenames(last_user_request)

    if len(mentioned) < 2:
        return None

    if not successfully_modified_paths:
        return None

    modified_basenames = {
        Path(p).name for p in successfully_modified_paths
    }

    missing = sorted(
        m for m in mentioned if Path(m).name not in modified_basenames
    )

    if not missing:
        return None

    missing_list = "\n".join(f"  - {m}" for m in missing)

    return (
        "The request named multiple files, but at least one of them "
        f"doesn't appear to have been modified yet:\n{missing_list}\n\n"
        "If the task genuinely requires changes there, make them "
        "before giving a final answer. If a listed file doesn't "
        "actually need changes for this task, say so explicitly in "
        "the answer instead of silently leaving it out."
    )


# -----------------------------------------------------------------------
# M7.12 -- scope creep detection
# -----------------------------------------------------------------------
#
# The mirror-image failure class to M7.11: instead of leaving a named
# file untouched, the agent modifies files beyond what was actually
# requested. This is the fuzziest of the four M7.9-12 checks -- lots of
# legitimate changes reasonably touch more than one file (a bug fix
# pulling in a related import, an accompanying test update), so a
# check that fires on "any extra file touched" would false-positive
# constantly on a 3B model.
#
# To keep precision high, this only fires when the user's own request
# contains an explicit scope-limiting phrase (SCOPE_LIMIT_KEYWORDS,
# e.g. "only change `app/config.py`") AND names at least one specific
# file. Without both signals, this returns None rather than guessing --
# same philosophy as M7.6's REWRITE_INTENT_KEYWORDS and M7.7's
# filename-based evidence requirement.
#
# Unlike M7.6/M7.7/M7.10, this does not automatically revert the extra
# file(s): by the time an explicit scope-limit violation is detected,
# other legitimate edits may have happened in between, and blindly
# reverting whatever else changed risks destroying real work. Instead
# this flags the problem and asks the agent to either revert the extra
# file(s) itself or justify why they were necessary.
#
# Practical note: M7.7's wrong-file redirect already reverts most
# everyday scope creep on its own, independent of scope language --
# whenever the request names a file that already exists in the
# project, M7.7 treats *any* edit to a different file as a likely
# misdirected edit and redirects it back, before this check ever runs.
# M7.12 mainly adds coverage for the case M7.7 can't: when the named
# file doesn't exist yet (it's being created this run), M7.7 has no
# project file to redirect an extra edit to.


def check_scope_creep(
    last_user_request: str,
    successfully_modified_paths: set,
):
    """Check whether a scope-limited request touched files beyond what
    it named.

    Returns None if the request didn't contain an explicit
    scope-limiting phrase, didn't name any specific file, or if every
    successfully modified path matches something the request named.
    Otherwise returns a message listing the out-of-scope files.
    """

    if not isinstance(last_user_request, str):
        return None

    lowered = last_user_request.lower()

    if not any(keyword in lowered for keyword in SCOPE_LIMIT_KEYWORDS):
        return None

    mentioned = extract_mentioned_filenames(last_user_request)

    if not mentioned:
        return None

    if not successfully_modified_paths:
        return None

    allowed_basenames = {Path(m).name for m in mentioned}

    extra = sorted(
        p for p in successfully_modified_paths
        if Path(p).name not in allowed_basenames
    )

    if not extra:
        return None

    extra_list = "\n".join(f"  - {p}" for p in extra)

    return (
        "The request asked for the change to be limited to specific "
        f"file(s), but this run also modified:\n{extra_list}\n\n"
        "If those changes were genuinely necessary to complete the "
        "task, explain why in the final answer. Otherwise, revert "
        "them so only the requested file(s) are changed."
    )


def looks_like_action_request(text: str) -> bool:
    """Heuristic for M7.15: the request asks for work on the project
    (an action verb, or a named file plus intent), as opposed to a
    question that can be answered from general knowledge."""

    if not isinstance(text, str) or not text.strip():
        return False

    lowered = text.lower()

    if any(
        re.search(rf"\b{verb}\b", lowered)
        for verb in ACTION_REQUEST_VERBS
    ):
        return True

    return bool(extract_mentioned_filenames(text)) and any(
        word in lowered
        for word in ("in", "the", "this", "that", "it")
    )


# -----------------------------------------------------------------------
# M7.8 -- false verification claim detection
# -----------------------------------------------------------------------
#
# M7.6 and M7.7 both catch a modification tool actually doing the wrong
# thing. This catches a different failure: the agent's own final answer
# asserting a verification step happened ("I verified this works",
# "tests pass") when no tool call capable of checking that was actually
# made since the last successful modification. The system prompt
# already tells the model not to do this; this is the enforcement.
#
# Deliberately scoped to the claim, not the underlying behavior: this
# does not force the agent to actually verify its work, only to not lie
# about having done so. A retry that rephrases the answer to drop the
# claim (without verifying anything) satisfies the check just as well
# as one that runs a real verification step -- which matches the
# failure class this targets ("false claims"), not "unverified changes
# are forbidden".


def claims_verification(text: str) -> bool:
    """Check whether a final answer asserts that verification happened."""

    if not isinstance(text, str):
        return False

    lowered = text.lower()

    return any(keyword in lowered for keyword in VERIFICATION_CLAIM_KEYWORDS)


# -----------------------------------------------------------------------
# M7.9 -- unread-file edit guard
# -----------------------------------------------------------------------
#
# The system prompt already says: "Before modifying an existing file,
# read it first unless its relevant contents are already available."
# Nothing enforced that. A model can call a modification tool on an
# existing file it has never read or searched this run, working
# from an assumption about the content instead of the actual content --
# which is exactly the kind of blind edit M7.6/M7.7 exist to catch
# after the fact. This catches it before the tool ever runs: unlike
# M7.6/M7.7/M7.8, whether the model has "seen" a given path is fully
# knowable in advance, so there's no need for a write-then-revert cycle
# here -- block it the same way check_unnecessary_full_rewrite() does.
#
# "Seen" means either read_file(path) or a search_files() call that
# returned at least one match in that path -- both actually expose real
# file content, unlike list_files() or git_status(). A path is also
# considered seen the moment a modification tool successfully touches
# it: the model wrote that content itself (write_file), or matched an
# exact substring of it (replace_in_file), so it isn't blind to a
# follow-up edit on the same path later in the same run. (A
# delete_file success removes the path, so there is no follow-up to
# unblock -- but the guard still applies to the deletion itself.)


def check_unread_file_edit(
    tool_name: str,
    arguments: dict,
    files_seen_this_run: set,
):
    """Guard run before executing a modification tool.

    If the target is an existing file that hasn't been read or searched
    at all this run, returns a corrective message instead of letting
    the modification proceed blind. Returns None if the call should
    proceed as-is (new file, already-seen file, or unusable arguments).
    """

    if tool_name not in MODIFICATION_TOOLS:
        return None

    if not isinstance(arguments, dict):
        return None

    path = arguments.get("path")

    if not isinstance(path, str) or not path.strip():
        return None

    if path in files_seen_this_run:
        return None

    if safe_read_file(path) is None:
        # New file (or unreadable) -- nothing to have read first.
        return None

    return (
        f"`{path}` already exists, but it hasn't been read or searched "
        "yet this run, so this modification would be made blind, "
        "without knowing the file's actual current content.\n\n"
        f"Call `read_file` (or `search_files`) on `{path}` first, then "
        "retry the modification based on what it actually contains."
    )
