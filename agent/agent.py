import json

import ollama

from agent.tools import TOOLS


MODEL = "qwen2.5-coder:3b"
MAX_TOOL_CALLS = 5

# Tools whose failure represents an incomplete modification, not just
# an information-gathering step. Used to detect and recover from
# failed edits (M7.3).
MODIFICATION_TOOLS = {"write_file", "replace_in_file"}

# How many times the agent is forced to keep trying after a genuine
# modification failure before it is allowed to give up. Bounds the
# recovery loop so it can't run forever.
MAX_RECOVERY_ATTEMPTS = 2

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


def run_agent(messages: list[dict]):
    """Run the agent until it produces a final answer."""

    tool_calls = 0

    # M7.3 recovery state: tracks whether the most recent modification
    # attempt failed for a genuine (recoverable) reason, and how many
    # times we've forced the agent to keep trying instead of quitting.
    pending_recovery = False
    recovery_attempts = 0

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
            print(f"\nAgent produced invalid JSON: {e}")
            return None

        tool_name = normalize_tool_name(
            action.get("tool")
        )

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
            print(f"\nUnknown tool requested: {tool_name}")
            return None

        arguments = action.get("arguments", {})

        print(f"\nTool requested: {tool_name}")
        print(f"Arguments: {arguments}")

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

        messages.append({
            "role": "assistant",
            "content": raw_response,
        })

        messages.append({
            "role": "user",
            "content": (
                f"Tool `{tool_name}` returned:\n"
                f"{json.dumps(result, indent=2)}\n\n"
                "Continue working. "
                "Use another tool if necessary, "
                "or return your final answer."
            ),
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