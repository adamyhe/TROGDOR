"""
pytest configuration for TROGDOR tests.

Markers:
    integration  — tests that require real BigWig/BED files on disk.
                   Run with: pytest -m integration
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring real data files (deselected by default)",
    )


def pytest_collection_modifyitems(config, items):
    # Skip integration tests unless explicitly selected
    if not config.option.markexpr or "integration" not in config.option.markexpr:
        skip_integration = pytest.mark.skip(reason="use -m integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
