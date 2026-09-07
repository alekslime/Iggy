import json

import ollama

from agent.tools import TOOLS


MODEL = "deepseek-r1:8b"
MAX_TOOL_CALLS = 5

client = ollama.Client(host="http://127.0.0.1:11434")


SYSTEM_PROMPT = """
You are a local computer engineering agent.

You have access to tools.

When you need to use a tool, respond with ONLY valid JSON:

{
    "tool": "tool_name",
    "arguments": {
        "argument": "value"
    }
}

When you can answer without a tool, respond with ONLY valid JSON:

{
    "tool": "none",
    "answer": "your answer"
}

Available tools:
- list_files(root): lists relevant files inside a project directory.

Rules:
- Do not invent tools.
- Do not put markdown around JSON.
- Use tools when they are useful.
- After receiving a tool result, decide what to do next.
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

        # Remove opening ``` or ```json
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Remove closing ```
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

    # DeepSeek may return:
    # list_files(root)
    #
    # instead of:
    # list_files
    if "(" in tool_name:
        tool_name = tool_name.split("(", 1)[0].strip()

    return tool_name


def run_agent(messages: list[dict]):
    """Run the agent until it produces a final answer."""

    tool_calls = 0

    while tool_calls < MAX_TOOL_CALLS:

        response = client.chat(
            model=MODEL,
            messages=messages,
        )

        raw_response = response.message.content

        print(f"\nDeepSeek raw response:\n{raw_response}")

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

        # Tell the model what it requested.
        messages.append({
            "role": "assistant",
            "content": raw_response,
        })

        # Give the model the tool result.
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
            print(f"\nDeepSeek: {answer}")

            messages.append({
                "role": "assistant",
                "content": answer,
            })


if __name__ == "__main__":
    main()