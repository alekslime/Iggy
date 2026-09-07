import ollama


MODEL = "deepseek-r1:8b"


def main():
    print("Local Computer Engineer")
    print(f"Model: {MODEL}")
    print("-" * 40)
    print("Type 'quit' to exit.\n")

    client = ollama.Client(host="http://127.0.0.1:11434")

    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        print("\nDeepSeek:")

        response = client.chat(
            model=MODEL,
            messages=messages,
        )

        assistant_message = response.message.content

        print(assistant_message)
        print()

        messages.append({
            "role": "assistant",
            "content": assistant_message,
        })


if __name__ == "__main__":
    main()