def test_custom_option(pytestconfig):
    visual_option = pytestconfig.getoption("--visual")
    assert visual_option is not None, "The --visual option should be available"

    visual_option = pytestconfig.getoption("--visual-accept-all")
    assert visual_option is not None, "The --visual-accept-all option should be available"

    visual_option = pytestconfig.getoption("--visual-forget-all")
    assert visual_option is not None, "The --visual-forget-all option should be available"
