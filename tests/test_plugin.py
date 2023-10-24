# content of tests/test_plugin.py


def test_custom_option(pytestconfig):
    custom_option_value = pytestconfig.getoption("--visual")
    assert custom_option_value is not None, "The --visual option should be set"
