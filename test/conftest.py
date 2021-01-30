"Define fixtures and other test helpers."

import pytest


def pytest_addoption(parser):
    parser.addoption("--elements-to-test", action="store", default="ALL")
    parser.addoption("--cells-to-test", action="store", default="ALL")


@pytest.fixture
def elements_to_test(request):
    data = request.config.getoption("--elements-to-test")
    print(data)
    if data == "ALL":
        return "ALL"
    return data.split(",")


@pytest.fixture
def cells_to_test(request):
    data = request.config.getoption("--cells-to-test")
    print(data)
    if data == "ALL":
        return "ALL"
    return data.split(",")
