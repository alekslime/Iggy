def request_permission(action: str) -> bool:
    """Ask the user for permission to perform an action."""

    print("\nIggy wants permission to:")
    print(f"  {action}")

    response = input("\nAllow? [y/N]: ").strip().lower()

    return response == "y"