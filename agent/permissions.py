def request_permission(action: str) -> bool:
    """Ask the user for permission to perform an action.

    Prints the requested action and prompts the user for a y/N
    confirmation, returning True only on an explicit "y".
    """

    print("\nIggy needs your permission to:")
    print(f"  {action}")

    response = input("\nProceed? [y/N]: ").strip().lower()

    return response == "y"
