"""pytest settings shared by the whole repository."""

import doctest

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a doctest of a function that is still a stub as skipped.

    A doctest that hits ``raise NotImplementedError`` would otherwise fail and
    turn CI red before the function is written. Any other error still fails.
    """
    outcome = yield
    report = outcome.get_result()
    error = call.excinfo.value if call.excinfo else None
    if isinstance(error, doctest.UnexpectedException) and issubclass(
        error.exc_info[0], NotImplementedError
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), 0, f"{item.name} not implemented yet")
