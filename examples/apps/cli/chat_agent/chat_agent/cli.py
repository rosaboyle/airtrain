"""
Command-line interface for the chat agent.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import config
from .main import ChatAgent
from .storage import ChatSession

app = typer.Typer(help="A personal chat agent using Together AI via AirTrain.")
console = Console()

# Define command names to prioritize over session IDs
COMMAND_NAMES = ["list", "info", "chat"]


def find_session_by_partial_id(partial_id: str) -> Optional[str]:
    """Find a session by partial ID match (prefix)."""
    if not partial_id:
        return None

    sessions = ChatSession.list_sessions()
    for session in sessions:
        if session["session_id"].startswith(partial_id):
            return session["session_id"]
    return None


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to continue (can be partial)"
    ),
    list_sessions_flag: bool = typer.Option(
        False, "--list", help="List all available chat sessions."
    ),
    info_flag: bool = typer.Option(
        False, "--info", help="Show information about the chat storage location."
    ),
    chat_flag: bool = typer.Option(
        False, "--chat", help="Start a chat session with the AI."
    ),
    continue_session: Optional[str] = typer.Option(
        None,
        "--continue",
        "-c",
        help="Continue an existing chat session (can be partial ID).",
    ),
    delete_session: Optional[int] = typer.Option(
        None,
        "--delete",
        help="Delete a chat session by its number in the list.",
    ),
    help_flag: bool = typer.Option(
        False, "--help", "-h", help="Show this message and exit.", is_eager=True
    ),
):
    """Run the chat agent CLI."""
    # Show help if explicitly requested
    if help_flag:
        console.print(ctx.get_help())
        return

    # Check if any of the option flags are set
    if list_sessions_flag:
        show_sessions_list()
        return

    if info_flag:
        show_storage_info()
        return

    if delete_session is not None:
        delete_session_by_number(delete_session)
        return

    if chat_flag or continue_session:
        # If continue_session is provided, try to find a match with a partial ID
        if continue_session:
            full_id = find_session_by_partial_id(continue_session)
            if full_id:
                continue_session = full_id
                console.print(f"Continuing session: [cyan]{full_id}[/cyan]")
            else:
                console.print(f"[red]Session '{continue_session}' not found[/red]")
                console.print("Use 'z --list' to see available sessions.")
                return

        _start_chat(session_id=continue_session)
        return

    # If no options are set, check if a session ID is provided
    if session_id:
        # Try to find a session with a matching prefix
        full_session_id = find_session_by_partial_id(session_id)
        if full_session_id:
            console.print(f"Continuing session: [cyan]{full_session_id}[/cyan]")
            # Don't call the chat function directly to avoid Typer's parameter handling
            _start_chat(session_id=full_session_id)
        else:
            console.print(f"[red]Session '{session_id}' not found[/red]")
            console.print("Use 'z --list' to see available sessions.")
    else:
        # Start a new chat session by default instead of showing help
        _start_chat(session_id=None)


def _start_chat(session_id: Optional[str] = None):
    """Start a chat session without Typer's parameter handling."""
    agent = ChatAgent(session_id=session_id)
    agent.chat()


def show_sessions_list():
    """List all available chat sessions."""
    sessions = ChatSession.list_sessions()

    if not sessions:
        console.print("[yellow]No chat sessions found.[/yellow]")
        return

    table = Table(title="Available Chat Sessions")
    table.add_column("#", style="white")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Messages", style="blue")
    table.add_column("Preview", style="yellow")

    for i, session in enumerate(sessions, 1):
        table.add_row(
            str(i),
            session["session_id"],
            session["created_at"],
            str(session["message_count"]),
            session.get("preview", "No preview available"),
        )

    console.print(table)


def delete_session_by_number(session_number: int):
    """Delete a chat session by its number in the list."""
    sessions = ChatSession.list_sessions()

    if not sessions:
        console.print("[yellow]No chat sessions found to delete.[/yellow]")
        return

    if session_number < 1 or session_number > len(sessions):
        console.print(f"[red]Invalid session number: {session_number}[/red]")
        console.print(f"Please choose a number between 1 and {len(sessions)}")
        return

    # Get the session ID for the given number
    session_id = sessions[session_number - 1]["session_id"]

    # Delete the session
    if ChatSession.delete_session(session_id):
        sid = session_id[:20] + "..." if len(session_id) > 23 else session_id
        msg = f"[green]Successfully deleted session {session_number}: {sid}[/green]"
        console.print(msg)
    else:
        sid = session_id[:20] + "..." if len(session_id) > 23 else session_id
        msg = f"[red]Failed to delete session {session_number}: {sid}[/red]"
        console.print(msg)


def show_storage_info():
    """Show information about the chat storage location."""
    console.print("Chat history storage location:")
    console.print(f"[green]{config.storage_dir}[/green]")

    # Check if the directory exists
    if config.storage_dir.exists():
        console.print("Directory exists: [green]Yes[/green]")
        sessions = ChatSession.list_sessions()
        console.print(f"Number of sessions: [blue]{len(sessions)}[/blue]")
    else:
        console.print("Directory exists: [red]No[/red]")
        console.print(
            "[yellow]The directory will be created when you start a chat.[/yellow]"
        )


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()