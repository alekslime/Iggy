import ast
import difflib
import json
import re
from pathlib import Path

import ollama

from agent.tools import (
    TOOLS,
    list_files,
    read_file,
    remove_created_file,
    restore_file_content,
)


MODEL = "qwen2.5-coder:3b"
MAX_TOOL_CALLS = 10

# Tools whose failure represents an incomplete modification, not just
# an information-gathering step. Used to detect and recover from
# failed edits (M7.3).
MODIFICATION_TOOLS = {"write_file", "replace_in_file"}

# How many times the agent is forced to keep trying after a genuine
# modification failure before it is allowed to give up. Bounds the
# recovery loop so it can't run forever.
MAX_RECOVERY_ATTEMPTS = 2

# How many consecutive malformed responses (invalid JSON, or a
# hallucinated/unknown tool name) the agent gets to correct itself
# before the run is aborted (M7.5). This is a different failure class
# from MAX_RECOVERY_ATTEMPTS: no real tool was ever executed here.
MAX_INVALID_ACTION_ATTEMPTS = 3

# How many times the agent is forced to retry after a modification that
# *succeeded* but failed semantic verification (M7.6) -- e.g. it wrote
# the requested text but also destroyed unrelated parts of the file.
# A different failure class again: the tool call itself worked fine.
MAX_SEMANTIC_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after a modification that
# *succeeded*, and passed semantic verification, but appears to have
# landed on the wrong file entirely (M7.7) -- e.g. the request named
# `utils.py` but the edit was made to `helpers/utils.py`, or the
# project has two files with that name and it's not clear which one
# was meant. Yet another distinct failure class: the edit itself may be
# perfectly correct, just applied to the wrong target.
MAX_WRONG_FILE_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after claiming, in its
# final answer, that a change was "verified" / "tested" / "confirmed"
# without actually calling a tool capable of checking that since the
# modification (M7.8). This is orthogonal to whether the modification
# itself was correct -- M7.6/M7.7 already gate that. This catches the
# agent asserting a verification step happened when it didn't, which
# the system prompt explicitly forbids ("Never claim that a change was
# verified unless you actually performed verification").
MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after a modification or
# file creation that succeeded, passed semantic and target-file
# verification, but left a .py file that doesn't parse as valid Python
# (M7.10). Checked regardless of whether the file already had a syntax
# error before the edit -- nothing Iggy touches or creates should exit
# a run non-parseable, existing problems included.
MAX_SYNTAX_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after trying to give a
# final answer while the request named multiple files and at least one
# of them hasn't received a successful, verified modification yet
# (M7.11). Recomputed live at each final-answer attempt rather than
# tracked as a "pending" flag like M7.6/M7.7/M7.10 -- the underlying
# signal (which files have actually been modified so far) is cheap to
# recheck and naturally resolves itself as the agent does more work, so
# there's no separate failure state to remember between turns.
MAX_INCOMPLETE_MULTI_TARGET_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after modifying a file
# beyond what an explicitly scope-limited request named (M7.12). Like
# M7.11, recomputed live rather than tracked as a pending flag.
MAX_SCOPE_CREEP_RETRY_ATTEMPTS = 2

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
# "agent/tools.py"). Restricting to known extensions -- rather than
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

client = ollama.Client(host="http://127.0.0.1:11434")


SYSTEM_PROMPT = """
You are a local computer engineering agent.

You have access to tools and are expected to ACT on tasks when the user asks you to modify the project.

When you need to use a tool, respond with ONLY valid JSON:

{
    "tool": "tool_name",
    "arguments": {
        "argument": "value"
    }
}

When you are completely finished and no more tools are needed, respond with ONLY valid JSON:

{
    "tool": "none",
    "answer": "your answer"
}

JSON formatting rules:
- Your entire response must be valid JSON.
- Never put literal newlines inside JSON string values.
- Use escaped newlines (\\n) inside strings when needed.
- Use double quotes for JSON strings.

IMPORTANT TASK RULES:

1. If the user asks you to INSPECT, CREATE, MODIFY, CHANGE, FIX, REMOVE, REFACTOR, IMPLEMENT, UPDATE, or otherwise alter something in the project, this is an ACTION REQUEST.

2. For an ACTION REQUEST:
   - Do not simply tell the user how to do it.
   - Do not return "none" before performing the requested work.
   - Inspect the relevant project files first.
   - Determine what needs to change.
   - Use the appropriate modification tool.
   - Continue working until the requested task is completed or you cannot safely complete it.

3. If the user asks for information about the project:
   - Use tools when necessary to inspect the project.
   - Return "none" only after you have enough information to answer.

4. After modifying something, verify the result when possible.

5. Never claim that a change was completed unless the modification tool actually succeeded.

6. Never claim that a change was verified unless you actually performed verification.

Before answering, ask yourself:

1. Is this an action request or an information request?
2. If it is an action request, what files do I need to inspect?
3. Have I inspected the relevant code?
4. What is the smallest safe change that completes the task?
5. Which modification tool should I use?
6. Has the modification actually succeeded?
7. Can I verify the result?
8. Only after completing the task should I return "none".

TOOL SELECTION:

- Use list_files when you need to discover the project structure.
- Use search_files when you need to find specific code, variables, functions, classes, imports, or text.
- Use read_file when you need to understand the contents of a file.
- Use replace_in_file when making a targeted change to existing code.
- Prefer replace_in_file over write_file when modifying an existing file.
- Use write_file when creating a new file or when replacing an entire file is genuinely necessary.
- Use run_command when a command is needed for inspection or verification.
- Use git_status when the user asks about Git status, modified files, untracked files, staged files, or changes in the working tree.

TOOL DETAILS:

- list_files(root): lists relevant files inside a project directory.
- read_file(path): reads a text file inside the project directory.
- search_files(query): searches project files for matching text and returns file names, line numbers, and matching lines.
- git_status(): returns the current Git working tree status.
- run_command(command): runs an approved command inside the project.
- write_file(path, content): writes or replaces a file inside the project after requesting user permission.
- replace_in_file(path, old_text, new_text): replaces one exact piece of existing text inside a project file after requesting user permission.

SAFETY AND MODIFICATION RULES:

- Never access files outside the project directory.
- Never invent directory names or paths.
- Tool arguments must refer to real paths or values supported by the tools.
- Before modifying an existing file, read it first unless its relevant contents are already available.
- Do not modify files blindly.
- Make the smallest change necessary to complete the user's request.
- Preserve existing code and behavior unless the user explicitly asks for a behavior change.
- When using replace_in_file, old_text must match exactly one occurrence.
- If the exact text does not exist, do not invent a replacement.
- If the exact text occurs more than once, do not make the change.
- If a modification is denied, do not pretend that it succeeded.
- If a tool returns an error, treat the operation as unsuccessful.
- Do not invent tools.
- Do not put markdown around JSON.
- After receiving a tool result, decide what to do next.

SEARCH RULES:

- For search_files, use short, exact keywords rather than descriptive phrases.
- For example, search for "ollama", not "Ollama usage".
- For list_files, use "." for the project root unless you have a specific directory name from a previous tool result.
- Never invent paths.

COMPLETION RULE:

For an action request, do not stop simply because you have figured out what should be changed.

The task is only complete after:
- the requested modification was successfully applied, OR
- the agent has determined that it cannot safely complete the task.

When possible, perform a verification step after the modification.

Distinguish clearly between:
- planned
- modified
- verified

Only report what actually happened.
"""


def execute_tool(tool_name: str, arguments: dict):
    """Execute a registered tool."""

    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = TOOLS[tool_name]

    return tool(**arguments)


def clean_model_response(raw_response: str) -> str:
    """Remove Markdown code fences from the model response."""

    clean_response = raw_response.strip()

    if clean_response.startswith("```"):
        lines = clean_response.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        clean_response = "\n".join(lines).strip()

    return clean_response


def parse_action(raw_response: str) -> dict:
    """Convert the model's response into a tool/action dictionary."""

    clean_response = clean_model_response(raw_response)

    action = json.loads(clean_response)

    if not isinstance(action, dict):
        raise ValueError("Model response must be a JSON object.")

    return action


def normalize_tool_name(tool_name):
    """Normalize tool names returned by the model."""

    if not isinstance(tool_name, str):
        return tool_name

    if "(" in tool_name:
        tool_name = tool_name.split("(", 1)[0].strip()

    return tool_name


def classify_modification_result(tool_name: str, result) -> str:
    """Classify the outcome of a modification tool call.

    Returns one of:
    - "success": the modification was actually applied.
    - "denied": the user declined permission (not a failure to retry).
    - "failed": a genuine, recoverable tool failure occurred.
    - "not_applicable": this tool call isn't a modification, or its
      result doesn't match a known pattern.

    This only inspects the exact strings write_file/replace_in_file
    are documented to return, plus the {"error": ...} shape produced
    when execute_tool raises. It does not guess at other cases.
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

    if result.startswith("Replacement failed:"):
        return "failed"

    return "not_applicable"


# -----------------------------------------------------------------------
# M7.6 -- semantic edit verification
# -----------------------------------------------------------------------
#
# The mechanisms above (M7.3/M7.5) only know whether a tool call itself
# succeeded or failed. They can't tell that a `write_file` call which
# reported success actually destroyed the file it was supposed to make
# one small edit to. The helpers below add two cheap, non-AST checks:
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
# bar is "nothing Iggy leaves behind is broken Python," not "don't make
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
# Nothing enforced that. A model can call write_file or replace_in_file
# on an existing file it has never read or searched this run, working
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
# follow-up edit on the same path later in the same run.


def check_unread_file_edit(
    tool_name: str,
    arguments: dict,
    files_seen_this_run: set,
):
    """Guard run before executing write_file/replace_in_file.

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


def run_agent(messages: list[dict]):
    """Run the agent until it produces a final answer."""

    tool_calls = 0

    # M7.3 recovery state: tracks whether the most recent modification
    # attempt failed for a genuine (recoverable) reason, and how many
    # times we've forced the agent to keep trying instead of quitting.
    pending_recovery = False
    recovery_attempts = 0

    # M7.5 recovery state: tracks consecutive malformed responses
    # (invalid JSON or an unknown/hallucinated tool name), so the
    # agent can be corrected instead of the run silently dying.
    invalid_action_attempts = 0

    # M7.6 recovery state: tracks whether the most recent modification
    # *succeeded* as a tool call but failed semantic verification (wrong
    # and/or destructive result), and how many times we've forced a
    # retry instead of letting the agent claim success.
    pending_semantic_recovery = False
    semantic_recovery_attempts = 0
    last_semantic_failure_message = None

    # M7.7 recovery state: tracks whether the most recent modification
    # succeeded and passed semantic verification, but appears to have
    # landed on the wrong file (name mismatch or unresolved ambiguity
    # against the user's request), and how many times we've forced a
    # retry on the correct file instead of letting the agent claim
    # success.
    pending_wrong_file_recovery = False
    wrong_file_recovery_attempts = 0
    last_wrong_file_failure_message = None

    # M7.10 recovery state: tracks whether the most recent modification
    # or file creation succeeded, passed semantic and target-file
    # verification, but left a .py file that doesn't parse, and how
    # many times we've forced a retry instead of letting the agent
    # claim success with broken syntax.
    pending_syntax_recovery = False
    syntax_recovery_attempts = 0
    last_syntax_failure_message = None

    # M7.11/M7.12 shared state: tracks which project-relative paths
    # have received a modification that passed *every* check this run
    # (M7.6 semantic, M7.7 target-file, M7.10 syntax) -- i.e. edits
    # that actually stuck, not ones that got reverted. Both checks are
    # recomputed live from this set at each final-answer attempt rather
    # than tracked as separate pending flags.
    successfully_modified_paths = set()
    incomplete_multi_target_retry_attempts = 0
    scope_creep_retry_attempts = 0

    # M7.8 recovery state: tracks whether the most recent *successful*
    # modification has gone unverified by any tool call since (no
    # run_command / read_file / search_files), so a final answer that
    # claims verification happened can be caught and forced to either
    # actually check, or stop claiming it did.
    unverified_modification_pending = False
    verification_claim_recovery_attempts = 0

    # M7.9 guard state: tracks which project-relative paths have
    # actually had their content exposed to the model this run, via
    # read_file, a matching search_files hit, or a successful
    # modification. Used to block write_file/replace_in_file calls
    # against existing files the model hasn't actually looked at.
    files_seen_this_run = set()

    # M7.6: the genuine user request driving this run, captured once up
    # front. Used to infer edit intent (small targeted change vs. full
    # rewrite). Deliberately captured before the loop starts so that the
    # corrective messages this loop injects as "user" turns don't get
    # mistaken for the actual task.
    last_user_request = ""

    for message in reversed(messages):
        if message.get("role") == "user":
            last_user_request = message.get("content", "")
            break

    while tool_calls < MAX_TOOL_CALLS:

        response = client.chat(
            model=MODEL,
            messages=messages,
        )

        raw_response = response.message.content

        print(f"\nModel raw response:\n{raw_response}")

        try:
            action = parse_action(raw_response)

        except (json.JSONDecodeError, ValueError) as e:

            invalid_action_attempts += 1

            print(f"\nAgent produced invalid JSON: {e}")

            if invalid_action_attempts > MAX_INVALID_ACTION_ATTEMPTS:
                print(
                    "\nAgent stopped after "
                    f"{MAX_INVALID_ACTION_ATTEMPTS} consecutive "
                    "invalid responses."
                )
                return None

            messages.append({
                "role": "assistant",
                "content": raw_response,
            })

            messages.append({
                "role": "user",
                "content": (
                    f"Your last response could not be parsed as JSON: "
                    f"{e}\n\n"
                    "Respond with ONLY a single valid JSON object in "
                    "the required format, and nothing else."
                ),
            })

            continue

        tool_name = normalize_tool_name(
            action.get("tool")
        )

        if tool_name == "none" or tool_name in TOOLS:
            invalid_action_attempts = 0

        # -----------------------------------------
        # FINAL ANSWER
        # -----------------------------------------

        if tool_name == "none":

            if pending_recovery and recovery_attempts < MAX_RECOVERY_ATTEMPTS:

                recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: a modification "
                    f"attempt failed and has not been recovered from "
                    f"(forced retry {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification attempt failed. This is a "
                        "recoverable tool failure, not a user permission "
                        "denial, so you may not give up yet.\n\n"
                        "Re-read or search the file to find the correct "
                        "exact text, then retry the modification with "
                        "corrected arguments.\n\n"
                        "Only respond with `tool: none` again if, after "
                        "actually retrying, you determine you genuinely "
                        "cannot safely complete the modification. In that "
                        "case, explain specifically what you tried and why "
                        "it cannot proceed."
                    ),
                })

                continue

            if (
                pending_semantic_recovery
                and semantic_recovery_attempts < MAX_SEMANTIC_RETRY_ATTEMPTS
            ):

                semantic_recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: the last "
                    "modification failed semantic verification (forced "
                    f"retry {semantic_recovery_attempts}/"
                    f"{MAX_SEMANTIC_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification technically succeeded as "
                        "a tool call, but did not pass verification, so "
                        "the task is not complete and the file has been "
                        "reverted:\n\n"
                        f"{last_semantic_failure_message}\n\n"
                        "Do not tell the user the task is complete. Fix "
                        "the issue above and retry the modification."
                    ),
                })

                continue

            if (
                pending_wrong_file_recovery
                and wrong_file_recovery_attempts < MAX_WRONG_FILE_RETRY_ATTEMPTS
            ):

                wrong_file_recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: the last "
                    "modification appears to have landed on the wrong "
                    f"file (forced retry {wrong_file_recovery_attempts}/"
                    f"{MAX_WRONG_FILE_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification succeeded and passed "
                        "content verification, but appears to have been "
                        "made to the wrong file, so the task is not "
                        "complete and the file has been reverted:\n\n"
                        f"{last_wrong_file_failure_message}\n\n"
                        "Do not tell the user the task is complete. Fix "
                        "the issue above and retry the modification on "
                        "the correct file."
                    ),
                })

                continue

            if (
                pending_syntax_recovery
                and syntax_recovery_attempts < MAX_SYNTAX_RETRY_ATTEMPTS
            ):

                syntax_recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: the last "
                    "modification left a file that doesn't parse "
                    f"(forced retry {syntax_recovery_attempts}/"
                    f"{MAX_SYNTAX_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification succeeded and passed "
                        "content and target-file verification, but left "
                        "the file with invalid syntax, so the task is "
                        "not complete:\n\n"
                        f"{last_syntax_failure_message}\n\n"
                        "Do not tell the user the task is complete. Fix "
                        "the syntax issue above and retry."
                    ),
                })

                continue

            if (
                incomplete_multi_target_retry_attempts
                < MAX_INCOMPLETE_MULTI_TARGET_RETRY_ATTEMPTS
            ):

                incomplete_message = check_incomplete_multi_target_edit(
                    last_user_request,
                    successfully_modified_paths,
                )

                if incomplete_message is not None:

                    incomplete_multi_target_retry_attempts += 1

                    print(
                        "\nRejecting premature final answer: the "
                        "request named multiple files and at least one "
                        "hasn't been modified yet (forced retry "
                        f"{incomplete_multi_target_retry_attempts}/"
                        f"{MAX_INCOMPLETE_MULTI_TARGET_RETRY_ATTEMPTS})."
                    )

                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                    messages.append({
                        "role": "user",
                        "content": incomplete_message,
                    })

                    continue

            if (
                scope_creep_retry_attempts
                < MAX_SCOPE_CREEP_RETRY_ATTEMPTS
            ):

                scope_creep_message = check_scope_creep(
                    last_user_request,
                    successfully_modified_paths,
                )

                if scope_creep_message is not None:

                    scope_creep_retry_attempts += 1

                    print(
                        "\nRejecting premature final answer: the "
                        "request was scope-limited but extra files "
                        "were modified (forced retry "
                        f"{scope_creep_retry_attempts}/"
                        f"{MAX_SCOPE_CREEP_RETRY_ATTEMPTS})."
                    )

                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                    messages.append({
                        "role": "user",
                        "content": scope_creep_message,
                    })

                    continue

            answer = action.get("answer", "")

            if (
                unverified_modification_pending
                and claims_verification(answer)
                and verification_claim_recovery_attempts
                < MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS
            ):

                verification_claim_recovery_attempts += 1

                print(
                    "\nRejecting final answer: it claims verification "
                    "that was never actually performed (forced retry "
                    f"{verification_claim_recovery_attempts}/"
                    f"{MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your answer claims the change was verified, "
                        "tested, or confirmed, but no verification step "
                        "(run_command, read_file, or search_files) has "
                        "actually been performed since the modification "
                        "was made.\n\n"
                        "Either actually verify the change now (e.g. "
                        "re-read the file, search for the change, or run "
                        "a relevant command), or give your answer again "
                        "without claiming a verification step that "
                        "didn't happen."
                    ),
                })

                continue

            messages.append({
                "role": "assistant",
                "content": raw_response,
            })

            return answer

        # -----------------------------------------
        # TOOL CALL
        # -----------------------------------------

        if tool_name not in TOOLS:

            invalid_action_attempts += 1

            print(f"\nUnknown tool requested: {tool_name}")

            if invalid_action_attempts > MAX_INVALID_ACTION_ATTEMPTS:
                print(
                    "\nAgent stopped after "
                    f"{MAX_INVALID_ACTION_ATTEMPTS} consecutive "
                    "invalid responses."
                )
                return None

            messages.append({
                "role": "assistant",
                "content": raw_response,
            })

            valid_tools = ", ".join(sorted(TOOLS) + ["none"])

            messages.append({
                "role": "user",
                "content": (
                    f'"{tool_name}" is not a real tool and cannot be '
                    "used.\n\n"
                    f"The only valid values for \"tool\" are: "
                    f"{valid_tools}.\n\n"
                    "Choose one of these, or use \"tool\": \"none\" if "
                    "you are ready to give your final answer."
                ),
            })

            continue

        arguments = action.get("arguments", {})

        print(f"\nTool requested: {tool_name}")
        print(f"Arguments: {arguments}")

        # M7.9: before letting any modification tool touch an existing
        # file, check whether the model has actually read or searched
        # that file at all this run. Applies to both write_file and
        # replace_in_file -- a blind edit is exactly as risky either
        # way. Checked first, ahead of M7.6's rewrite guard, since
        # there's no point steering a blind edit towards a "better"
        # tool before the model has even looked at the file.
        if tool_name in MODIFICATION_TOOLS:

            unread_file_guard_message = check_unread_file_edit(
                tool_name,
                arguments,
                files_seen_this_run,
            )

            if unread_file_guard_message is not None:

                print(
                    "\nBlocking modification: target file hasn't been "
                    "read or searched yet this run."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": unread_file_guard_message,
                })

                tool_calls += 1

                continue

        # M7.6: before letting an existing file be fully replaced, check
        # whether the request actually described a small, targeted edit
        # that replace_in_file should handle instead. If so, block the
        # write_file call before it ever touches disk.
        if tool_name == "write_file":

            rewrite_guard_message = check_unnecessary_full_rewrite(
                arguments,
                last_user_request,
            )

            if rewrite_guard_message is not None:

                print(
                    "\nBlocking write_file: looks like an unnecessary "
                    "full-file rewrite of a small edit."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": rewrite_guard_message,
                })

                tool_calls += 1

                continue

        # M7.6: capture the file's content immediately before a
        # modification tool runs, so it can be compared against the
        # result afterwards (and restored if verification fails).
        modification_path = (
            arguments.get("path")
            if tool_name in MODIFICATION_TOOLS and isinstance(arguments, dict)
            else None
        )
        pre_content = (
            safe_read_file(modification_path)
            if modification_path is not None
            else None
        )

        try:
            result = execute_tool(
                tool_name,
                arguments,
            )

        except Exception as e:
            result = {
                "error": str(e)
            }

        print("\nTool executed:")
        print(result)

        # M7.9: record that this path's real content has now been
        # exposed to the model this run, so a later modification
        # against the same path isn't blocked as blind.
        if tool_name == "read_file" and not (
            isinstance(result, dict) and "error" in result
        ):

            read_path = (
                arguments.get("path") if isinstance(arguments, dict) else None
            )

            if isinstance(read_path, str) and read_path.strip():
                files_seen_this_run.add(read_path)

        elif tool_name == "search_files" and isinstance(result, list):

            for match in result:
                if isinstance(match, dict) and isinstance(
                    match.get("file"), str
                ):
                    files_seen_this_run.add(match["file"])

        outcome = classify_modification_result(tool_name, result)

        if outcome == "failed":
            pending_recovery = True
        elif outcome in ("success", "denied"):
            pending_recovery = False
            recovery_attempts = 0

        # M7.9: a successful modification also counts as having "seen"
        # the file -- the model either wrote that content itself
        # (write_file) or matched an exact substring of it
        # (replace_in_file) -- so a follow-up edit to the same path
        # later in this run isn't blocked as blind.
        if outcome == "success" and modification_path is not None:
            files_seen_this_run.add(modification_path)

        # M7.8: track whether a successful modification still hasn't
        # been checked by any tool call since. Independent of whether
        # M7.6/M7.7 end up flagging this same modification -- those
        # gate the "none" response on their own, so it doesn't matter
        # if this stays set through a revert; the model can't reach a
        # verification-claim check until it gets past those first.
        if outcome == "success":
            unverified_modification_pending = True
            verification_claim_recovery_attempts = 0
        elif tool_name in VERIFICATION_TOOLS:
            unverified_modification_pending = False

        # M7.6: a tool call can report success while still not having
        # done what was asked (wrote the right text but destroyed the
        # rest of the file, etc). Only meaningful when we know what the
        # file looked like both before and after.
        #
        # M7.7: separately, a tool call can report success, write
        # perfectly consistent content, and still have landed on the
        # wrong file. Checked second (only if M7.6 didn't already flag
        # a content problem) since "right file, wrong content" and
        # "wrong file entirely" are different failure classes with
        # their own recovery messages; a single edit only triggers one.
        semantic_failure_message = None
        wrong_file_failure_message = None
        syntax_failure_message = None

        post_content = (
            safe_read_file(modification_path)
            if outcome == "success" and modification_path is not None
            else None
        )

        if pre_content is not None and post_content is not None:

            semantic_failure_message = verify_semantic_edit(
                path=modification_path,
                pre_content=pre_content,
                post_content=post_content,
                last_user_request=last_user_request,
            )

            if semantic_failure_message is None:
                wrong_file_failure_message = verify_target_file(
                    path=modification_path,
                    last_user_request=last_user_request,
                )

        # M7.10: checked whenever a modification/creation succeeded and
        # nothing above already flagged it -- covers both edits to
        # existing files (post_content already fetched above) and
        # brand-new files (pre_content is None, but post_content is
        # still the file's freshly-written content).
        if (
            outcome == "success"
            and post_content is not None
            and semantic_failure_message is None
            and wrong_file_failure_message is None
        ):
            syntax_failure_message = verify_python_syntax(
                modification_path,
                post_content,
            )

        if (
            semantic_failure_message is not None
            or wrong_file_failure_message is not None
            or syntax_failure_message is not None
        ):

            try:
                if pre_content is not None:
                    restore_file_content(modification_path, pre_content)
                    print(
                        f"\nVerification failed for {modification_path}; "
                        "reverted to its pre-edit content."
                    )
                else:
                    remove_created_file(modification_path)
                    print(
                        f"\nVerification failed for newly created "
                        f"{modification_path}; removed it."
                    )
            except Exception as e:
                print(
                    f"\nWarning: could not revert {modification_path} "
                    f"after failed verification: {e}"
                )

        if semantic_failure_message is not None:
            pending_semantic_recovery = True
            last_semantic_failure_message = semantic_failure_message
        elif outcome == "success":
            pending_semantic_recovery = False
            semantic_recovery_attempts = 0
            last_semantic_failure_message = None

        if wrong_file_failure_message is not None:
            pending_wrong_file_recovery = True
            last_wrong_file_failure_message = wrong_file_failure_message
        elif outcome == "success" and semantic_failure_message is None:
            pending_wrong_file_recovery = False
            wrong_file_recovery_attempts = 0
            last_wrong_file_failure_message = None

        if syntax_failure_message is not None:
            pending_syntax_recovery = True
            last_syntax_failure_message = syntax_failure_message
        elif (
            outcome == "success"
            and semantic_failure_message is None
            and wrong_file_failure_message is None
        ):
            pending_syntax_recovery = False
            syntax_recovery_attempts = 0
            last_syntax_failure_message = None

        # M7.11/M7.12: a modification only counts as "done" for the
        # multi-target-completeness and scope-creep checks once it has
        # cleared every other check above -- a reverted or removed
        # edit was never actually applied, so it shouldn't count either
        # towards "this file was handled" or towards "this file was
        # touched beyond what was asked."
        if (
            outcome == "success"
            and modification_path is not None
            and semantic_failure_message is None
            and wrong_file_failure_message is None
            and syntax_failure_message is None
        ):
            successfully_modified_paths.add(modification_path)

        messages.append({
            "role": "assistant",
            "content": raw_response,
        })

        if semantic_failure_message is not None:
            followup_content = (
                f"Tool `{tool_name}` reported success, but automatic "
                f"verification found a problem:\n\n"
                f"{semantic_failure_message}\n\n"
                "The file has been reverted to its state before this "
                "change. Do not tell the user the task is complete. Fix "
                "the issue above and try again."
            )
        elif wrong_file_failure_message is not None:
            followup_content = (
                f"Tool `{tool_name}` reported success, and the content "
                "looked correct, but automatic verification found a "
                "problem with which file was modified:\n\n"
                f"{wrong_file_failure_message}\n\n"
                "The file has been reverted to its state before this "
                "change. Do not tell the user the task is complete. Fix "
                "the issue above and try again."
            )
        elif syntax_failure_message is not None:
            revert_note = (
                "reverted to its state before this change"
                if pre_content is not None
                else "removed, since it was newly created"
            )
            followup_content = (
                f"Tool `{tool_name}` reported success, and content/target "
                "verification passed, but automatic syntax checking found "
                f"a problem:\n\n"
                f"{syntax_failure_message}\n\n"
                f"The file has been {revert_note}. Do not tell the user "
                "the task is complete. Fix the syntax issue above and "
                "try again."
            )
        else:
            followup_content = (
                f"Tool `{tool_name}` returned:\n"
                f"{json.dumps(result, indent=2)}\n\n"
                "Continue working. "
                "Use another tool if necessary, "
                "or return your final answer."
            )

        messages.append({
            "role": "user",
            "content": followup_content,
        })

        tool_calls += 1

    print(
        f"\nAgent stopped after reaching the "
        f"{MAX_TOOL_CALLS} tool-call limit."
    )

    return None


def main():

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("Local Computer Engineer")
    print(f"Model: {MODEL}")
    print("----------------------------------------")
    print("Type 'quit' to exit.")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        answer = run_agent(messages)

        if answer is not None:
            print(f"\nIggy: {answer}")


if __name__ == "__main__":
    main()