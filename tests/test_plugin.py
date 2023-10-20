# content of tests/test_plugin.py


def test_custom_option(pytestconfig):
    custom_option_value = pytestconfig.getoption("--visualize")
    assert custom_option_value is not None, "The --visualize option should be set"
