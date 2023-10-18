# content of tests/test_plugin.py

def test_custom_option(pytestconfig):
    custom_option_value = pytestconfig.getoption("--custom_option")
    assert custom_option_value is not None, "The custom option should be set"
