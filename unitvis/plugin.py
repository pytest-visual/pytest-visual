
def pytest_addoption(parser) -> None:
    parser.addoption("--custom_option", action="store", default="default_value")
    
def pytest_configure(config):
    custom_option_value = config.getoption("--custom_option")
    config._custom_option = custom_option_value
