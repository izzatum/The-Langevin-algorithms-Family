"""Keep small numerical tests reproducible and avoid CPU oversubscription."""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)
