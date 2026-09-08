import difflib
import json
import re

import ollama

from agent.tools import TOOLS, read_file, restore_file_content


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

# Fraction of a file's original content that must still be recognizable
# afterwards for a modification to NOT be considered destructive, used
# by content_preserved_ratio(). Two thresholds: edits with a clearly
# identified small target ("from X to Y") are held to a much stricter
# bar than edits where we couldn't extract an explicit target, since we
# have stronger evidence the change was supposed to be tiny.
TARGETED_EDIT_PRESERVATION_THRESHOLD = 0.7
GENERIC_PRESERVATION_THRESHOLD = 0.5

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

# Matches the common "change/update ... from `X` to `Y`" phrasing used
# for small, targeted textual edits. Deliberately simple: this is a
# practical heuristic for M7.6, not a natural-language parser. `X`/`Y`
# may be quoted with backticks, single, or double quotes.
CHANGE_PAIR_PATTERN = re.compile(
    r"from\s+[`'\"]([^`'\"]+)[`'\"]\s+to\s+[`'\"]([^`'\"]+)[`'\"]",
    re.IGNORECASE | re.DOTALL,
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
    """Best-effort extraction of an explicit edit target from a user
    request, e.g. "change the comment from `A` to `B`" -> ("A", "B").

    Returns (old_text, new_text) or None. This only recognizes one
    common phrasing ("from X to Y") -- it's a practical heuristic, not
    a natural-language parser, and is expected to return None for most
    requests that don't spell out the exact before/after text.
    """

    if not isinstance(text, str):
        return None

    match = CHANGE_PAIR_PATTERN.search(text)

    if not match:
        return None

    old_text, new_text = match.group(1).strip(), match.group(2).strip()

    if not old_text or not new_text or old_text == new_text:
        return None

    return old_text, new_text


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

        threshold = (
            TARGETED_EDIT_PRESERVATION_THRESHOLD
            if pair is not None
            else GENERIC_PRESERVATION_THRESHOLD
        )

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

            answer = action.get("answer", "")

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

        outcome = classify_modification_result(tool_name, result)

        if outcome == "failed":
            pending_recovery = True
        elif outcome in ("success", "denied"):
            pending_recovery = False
            recovery_attempts = 0

        # M7.6: a tool call can report success while still not having
        # done what was asked (wrote the right text but destroyed the
        # rest of the file, etc). Only meaningful when we know what the
        # file looked like both before and after.
        semantic_failure_message = None

        if outcome == "success" and pre_content is not None:

            post_content = safe_read_file(modification_path)

            if post_content is not None:
                semantic_failure_message = verify_semantic_edit(
                    path=modification_path,
                    pre_content=pre_content,
                    post_content=post_content,
                    last_user_request=last_user_request,
                )

        if semantic_failure_message is not None:

            try:
                restore_file_content(modification_path, pre_content)
                print(
                    f"\nSemantic verification failed for "
                    f"{modification_path}; reverted to its pre-edit "
                    "content."
                )
            except Exception as e:
                print(
                    f"\nWarning: could not revert {modification_path} "
                    f"after failed verification: {e}"
                )

            pending_semantic_recovery = True
            last_semantic_failure_message = semantic_failure_message

        elif outcome == "success":
            pending_semantic_recovery = False
            semantic_recovery_attempts = 0
            last_semantic_failure_message = None

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