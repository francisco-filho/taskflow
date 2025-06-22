import pytest


@pytest.fixture()
def fix1():
    return 1

def test_hello(fix1):
    assert 1 == fix1

