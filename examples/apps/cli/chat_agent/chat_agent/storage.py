"""
Storage management for chat history.
"""

import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from .config import config


class ChatSession:
    """Represents a chat session with messages history."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize a chat session with optional ID."""
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.messages: List[Dict[str, str]] = []
        self.file_path = config.storage_dir / f"{self.session_id}.json"

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(message)
        self.save()

    def save(self) -> None:
        """Save the chat session to disk."""
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": self.messages,
        }

        # Ensure directory exists
        config.storage_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, session_id: str) -> Optional["ChatSession"]:
        """Load a chat session from disk."""
        file_path = config.storage_dir / f"{session_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            session = cls(session_id=data["session_id"])
            session.created_at = data.get("created_at", datetime.now().isoformat())
            session.messages = data.get("messages", [])
            return session
        except (json.JSONDecodeError, KeyError):
            return None

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """Delete a chat session by its ID."""
        file_path = config.storage_dir / f"{session_id}.json"

        if not file_path.exists():
            return False

        try:
            os.remove(file_path)
            return True
        except (IOError, OSError):
            return False

    @classmethod
    def list_sessions(cls) -> List[Dict[str, Any]]:
        """List all available chat sessions with metadata."""
        sessions = []

        for file_path in config.get_sessions_list():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                session_info = {
                    "session_id": data["session_id"],
                    "created_at": data.get("created_at", "Unknown"),
                    "message_count": len(data.get("messages", [])),
                    "file_path": str(file_path),
                }

                # Add a preview of the first exchange if available
                messages = data.get("messages", [])
                if messages:
                    first_user_msg = next(
                        (m for m in messages if m["role"] == "user"), None
                    )
                    session_info["preview"] = (
                        first_user_msg["content"][:50] + "..."
                        if first_user_msg
                        else "No preview"
                    )

                sessions.append(session_info)
            except (json.JSONDecodeError, KeyError, IOError):
                # Skip invalid files
                continue

        return sessions
