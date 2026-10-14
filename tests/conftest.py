import socket
from pathlib import Path

import pytest
import yaml

from para_agent.config import PROJECT_ROOT, load_config
from para_agent.kb.lookup import KnowledgeBase


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Unit tests must never open network connections")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def cfg():
    return load_config(PROJECT_ROOT / "configs/default.yaml")


@pytest.fixture
def contracts():
    return yaml.safe_load((PROJECT_ROOT / "configs/contracts.yaml").read_text())


@pytest.fixture
def kb():
    return KnowledgeBase(PROJECT_ROOT / "data/kb/features.jsonl",
                         PROJECT_ROOT / "configs/feature_key_map.yaml")

