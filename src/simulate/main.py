def hello(name: str = "World") -> str:
    """Say hello."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(hello())  # noqa: T201
