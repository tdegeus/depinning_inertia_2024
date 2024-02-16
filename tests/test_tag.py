from depinning_inertia_2024 import tag


def test_has_uncomitted():
    assert tag.has_uncommitted("4.4.dev1+hash.bash")
    assert not tag.has_uncommitted("4.4.dev1+hash")
    assert not tag.has_uncommitted("4.4.dev1")
    assert not tag.has_uncommitted("4.4")


def test_any_has_uncommitted():
    m = "main=3.2.1"
    o = "other"

    assert tag.any_has_uncommitted([m, f"{o}=4.4.dev1+hash.bash"])
    assert not tag.any_has_uncommitted([m, f"{o}=4.4.dev1+hash"])
    assert not tag.any_has_uncommitted([m, f"{o}=4.4.dev1"])
    assert not tag.any_has_uncommitted([m, f"{o}=4.4"])


def test_greater_equal():
    assert not tag.greater_equal("4.4.dev1+hash.bash", "4.4")
    assert not tag.greater_equal("4.4.dev1+hash", "4.4")
    assert not tag.greater_equal("4.4.dev1", "4.4")
    assert tag.greater_equal("4.4", "4.4")


def test_greater():
    assert not tag.greater("4.4.dev1+hash.bash", "4.4")
    assert not tag.greater("4.4.dev1+hash", "4.4")
    assert not tag.greater("4.4.dev1", "4.4")
    assert not tag.greater("4.4", "4.4")


def test_less_equal():
    assert tag.less_equal("4.4.dev1+hash.bash", "4.4")
    assert tag.less_equal("4.4.dev1+hash", "4.4")
    assert tag.less_equal("4.4.dev1", "4.4")
    assert tag.less_equal("4.4", "4.4")


def test_less():
    assert tag.less("4.4.dev1+hash.bash", "4.4")
    assert tag.less("4.4.dev1+hash", "4.4")
    assert tag.less("4.4.dev1", "4.4")
    assert not tag.less("4.4", "4.4")


def test_all_greater_equal():
    a = ["main=3.2.1", "other=4.4"]
    b = ["main=3.2.0", "other=4.4", "more=3.0.0"]
    assert tag.all_greater_equal(a, b)

    a = ["main=3.2.1", "other=4.4"]
    b = ["main=3.2.1.dev1", "other=4.4", "more=3.0.0"]
    assert tag.all_greater_equal(a, b)

    a = ["main=3.2.1", "other=4.4"]
    b = ["main=3.2.1.dev1+g423e6a8", "other=4.4", "more=3.0.0"]
    assert tag.all_greater_equal(a, b)

    a = ["main=3.2.1", "other=4.4"]
    b = ["main=3.2.1.dev1+g423e6a8.d20210902", "other=4.4", "more=3.0.0"]
    assert tag.all_greater_equal(a, b)
