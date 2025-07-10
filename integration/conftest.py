# conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--score_uri", action="store", help="Endpoint scoring URI")
    parser.addoption("--score_key", action="store", help="Endpoint key")
    parser.addoption("--threshold", action="store", default=13.0, type=float, help="Anomaly threshold")
    parser.addoption("--include_visualizations", action="store_true", default=False, help="Include visualizations in response")


@pytest.fixture
def score_uri(request):
    """Fixture to provide the scoring URL"""
    return request.config.getoption("--score_uri")


@pytest.fixture
def score_key(request):
    """Fixture to provide the scoring key"""
    return request.config.getoption("--score_key")


@pytest.fixture
def score_uri(request):
    return request.config.getoption("--score_uri")

@pytest.fixture
def score_key(request):
    return request.config.getoption("--score_key")

@pytest.fixture
def threshold(request):
    return request.config.getoption("--threshold")

@pytest.fixture
def include_visualizations(request):
    return request.config.getoption("--include_visualizations")
