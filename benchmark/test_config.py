"""Hooks for ``benchopt test`` (see benchopt's test_config mechanism)."""

import pytest

# The real dataset needs the neuro stack and a large download, which CI does
# not carry; ``Simulated`` covers the no-data smoke test.
REAL_DATASETS = ("emg2pose",)


def check_test_dataset_install(dataset_class):
    if dataset_class.name.lower() in REAL_DATASETS:
        pytest.skip("real data is staged separately, not available in CI")


def check_test_dataset_get_data(dataset_class):
    if dataset_class.name.lower() in REAL_DATASETS:
        pytest.skip("real data is staged separately, not available in CI")
