tests/test_modules.py:
import pytest

def test_hello_world():
    assert 1 + 1 == 2

pytest.ini:
[pytest]
testpaths = tests
python_files = test_*.py