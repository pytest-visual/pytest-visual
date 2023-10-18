import pytest

from unitvis.visualize import Visualize

def pytest_addoption(parser):
    parser.addoption("--visualize", action="store_true", help="Run visualization tests")
    
@pytest.fixture
def visualize(request):
    if request.config.getoption("--visualize"):
        return Visualize()
    else:
        pytest.skip("Visualization is not enabled, add --visualize option to enable")
