def test_custom_option(pytestconfig):
    visual_option = pytestconfig.getoption("--visual")
    assert visual_option is not None, "The --visual option should be available"

    visual_option = pytestconfig.getoption("--visual-yes-all")
    assert visual_option is not None, "The --visual-yes-all option should be available"

    visual_option = pytestconfig.getoption("--visual-reset-all")
    assert visual_option is not None, "The --visual-reset-all option should be available"
