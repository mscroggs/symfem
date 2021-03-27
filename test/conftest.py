"""Define fixtures and other test helpers."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--elements-to-test", action="store", default="ALL")
    parser.addoption("--cells-to-test", action="store", default="ALL")
    parser.addoption("--has-basix", action="store", default="0")
    parser.addoption("--speed", action="store", default="slow")


@pytest.fixture
def elements_to_test(request):
    data = request.config.getoption("--elements-to-test")
    if data == "ALL":
        return "ALL"
    return data.split(",")


@pytest.fixture
def speed(request):
    return request.config.getoption("--speed")


@pytest.fixture
def cells_to_test(request):
    data = request.config.getoption("--cells-to-test")
    if data == "ALL":
        return "ALL"
    return data.split(",")


@pytest.fixture
def has_basix(request):
    data = request.config.getoption("--has-basix")
    return data == "1"
