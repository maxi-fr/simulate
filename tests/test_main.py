from simulate.main import hello


def test_hello() -> None:
    assert hello("Test") == "Hello, Test!"


def test_hello_default() -> None:
    assert hello() == "Hello, World!"
