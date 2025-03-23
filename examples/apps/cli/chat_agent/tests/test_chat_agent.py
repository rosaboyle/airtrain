"""
Tests for the chat agent CLI.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from chat_agent.config import Config
from chat_agent.storage import ChatSession


def test_config_default_storage_dir():
    """Test that the default storage directory is set correctly."""
    with patch.dict(os.environ, {}, clear=True):
        config = Config()
        assert config.storage_dir.parts[-3:] == (
            ".localagents",
            "chat_agent",
            "messages",
        )


def test_config_custom_storage_dir():
    """Test that a custom storage directory can be set via environment variable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict(os.environ, {"LOCALAGENTS_DIR": temp_dir}, clear=True):
            config = Config()
            assert str(config.storage_dir).startswith(temp_dir)
            assert config.storage_dir.parts[-2:] == ("chat_agent", "messages")


def test_chat_session_initialization():
    """Test that a chat session is initialized correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("chat_agent.storage.config") as mock_config:
            mock_config.storage_dir = Path(temp_dir)
            session = ChatSession()
            assert session.session_id is not None
            assert session.messages == []


def test_chat_session_add_message():
    """Test adding a message to a chat session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("chat_agent.storage.config") as mock_config:
            mock_config.storage_dir = Path(temp_dir)
            session = ChatSession()
            session.add_message("user", "Hello, AI!")

            assert len(session.messages) == 1
            assert session.messages[0]["role"] == "user"
            assert session.messages[0]["content"] == "Hello, AI!"
