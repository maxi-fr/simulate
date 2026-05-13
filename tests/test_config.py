from simulate.config import deep_merge


def test_deep_merge_simple() -> None:
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 3, "z": 4}}
    result = deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3}


def test_deep_merge_lists() -> None:
    base = {"a": [1, 2], "b": 3}
    override = {"a": [3, 4, 5]}
    result = deep_merge(base, override)
    assert result == {"a": [3, 4, 5], "b": 3}


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    result = deep_merge(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}
    assert result == {"a": {"x": 1, "y": 2}}
