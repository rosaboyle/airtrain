import pytest
import logging
import os


# Set up logging before tests run
def pytest_configure(config):
    """Configure pytest for more verbose output."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Add verbose flag if not already set
    if not config.option.verbose:
        config.option.verbose = 1

    # Add showlocals if not already set
    if not hasattr(config.option, "showlocals") or not config.option.showlocals:
        config.option.showlocals = True


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add additional information to the test report summary."""
    if exitstatus != 0:
        terminalreporter.section("Failed Tests Summary")
        for failed in terminalreporter.stats.get("failed", []):
            terminalreporter.line(f"FAILED: {failed.nodeid}")

            # Print the error message again for clarity
            if hasattr(failed, "longrepr"):
                terminalreporter.line(f"Error: {str(failed.longrepr)[:200]}...")


# Define a hook to print more information for each test
def pytest_runtest_protocol(item, nextitem):
    """Print extra information about each test."""
    print(f"\n{'='*80}")
    print(f"Running test: {item.nodeid}")
    print(f"{'='*80}\n")
    return None  # Continue with normal test execution
