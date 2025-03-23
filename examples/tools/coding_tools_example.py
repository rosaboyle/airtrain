#!/usr/bin/env python3
"""
Example usage of AirTrain's coding-related tools.

This script demonstrates how to use the tools for code navigation,
searching, and testing within a codebase.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

# Import required modules
from airtrain.tools import (
    ToolFactory,
    ListDirectoryTool,
    DirectoryTreeTool,
    ExecuteCommandTool,
    FindFilesTool,
    TerminalNavigationTool,
    SearchTermTool,
    RunPytestTool,
)


def demo_list_directory():
    """Demonstrate the ListDirectoryTool."""
    print("\n=== Listing Directory ===")
    tool = ToolFactory.get_tool("list_directory")
    result = tool(path=".", show_hidden=False)
    print(result)


def demo_directory_tree():
    """Demonstrate the DirectoryTreeTool."""
    print("\n=== Directory Tree ===")
    tool = ToolFactory.get_tool("directory_tree")
    result = tool(path=".", max_depth=2, show_hidden=False)
    print(result)


def demo_terminal_navigation():
    """Demonstrate the TerminalNavigationTool."""
    print("\n=== Terminal Navigation ===")
    nav = ToolFactory.get_tool("terminal_navigation", "stateful")

    # Show current directory
    result = nav(action="pwd")
    print(f"Current directory: {result['current_dir']}")

    # Change to parent directory
    result = nav(action="cd", directory="..")
    print(f"Changed to parent directory: {result['current_dir']}")

    # Push directory onto stack
    result = nav(action="pushd", directory="airtrain")
    if result["success"]:
        print(f"Pushed directory onto stack, now in: {result['current_dir']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    # Show directory stack
    result = nav(action="dirs")
    print(f"Directory stack: {result['dir_stack']}")

    # Pop directory from stack
    result = nav(action="popd")
    if result["success"]:
        print(f"Popped directory from stack, now in: {result['current_dir']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def demo_execute_command():
    """Demonstrate the ExecuteCommandTool."""
    print("\n=== Execute Command ===")
    tool = ToolFactory.get_tool("execute_command")

    # Run a simple command
    result = tool(command="echo 'Hello from AirTrain Tools!'")
    print(f"Command output: {result['stdout'].strip()}")

    # Run a command with working directory
    result = tool(command="ls -la", working_dir=".")
    print(f"Files in directory (first 3 lines):")
    for line in result["stdout"].splitlines()[:3]:
        print(f"  {line}")


def demo_find_files():
    """Demonstrate the FindFilesTool."""
    print("\n=== Find Files ===")
    tool = ToolFactory.get_tool("find_files")

    # Find Python files
    result = tool(directory=".", pattern="**/*.py", max_results=5)

    if result["success"]:
        print(f"Found {result['count']} Python files (showing first 5):")
        for file in result["files"]:
            print(f"  {file['path']}")

        if result["truncated"]:
            print("  (results truncated)")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def demo_search_term():
    """Demonstrate the SearchTermTool."""
    print("\n=== Search Term ===")
    tool = ToolFactory.get_tool("search_term")

    # Search for a term in Python files
    result = tool(term="BaseTool", directory=".", file_pattern="*.py", max_results=3)

    if result["success"]:
        print(
            f"Found {result['match_count']} matches for 'BaseTool' (showing first 3):"
        )
        for match in result["matches"]:
            print(f"  {match['file']}:{match['line']} - {match['content']}")

        if result["truncated"]:
            print("  (results truncated)")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def demo_run_pytest():
    """Demonstrate the RunPytestTool (if pytest is available)."""
    print("\n=== Run Pytest ===")
    tool = ToolFactory.get_tool("run_pytest")

    # Check if we're in a directory with tests
    find_tool = ToolFactory.get_tool("find_files")
    tests = find_tool(directory=".", pattern="**/test_*.py", max_results=1)

    if tests["success"] and tests["files"]:
        test_file = tests["files"][0]["path"]
        print(f"Running pytest on {test_file}")

        result = tool(test_path=test_file, verbose=True, timeout=10)

        if result["success"]:
            print("Tests passed!")
            if "summary" in result:
                for line in result["summary"]:
                    print(f"  {line}")
        else:
            if "return_code" in result and result["return_code"] != 0:
                print("Tests failed or encountered errors.")
                if "summary" in result:
                    for line in result["summary"]:
                        print(f"  {line}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print("No test files found. Skipping pytest demo.")


def main():
    """Run the coding tools demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate AirTrain coding tools")
    parser.add_argument(
        "--tool",
        choices=[
            "all",
            "list-dir",
            "dir-tree",
            "terminal-nav",
            "execute-cmd",
            "find-files",
            "search-term",
            "run-pytest",
        ],
        default="all",
        help="Which tool demo to run",
    )

    args = parser.parse_args()

    if args.tool in ["all", "list-dir"]:
        demo_list_directory()

    if args.tool in ["all", "dir-tree"]:
        demo_directory_tree()

    if args.tool in ["all", "terminal-nav"]:
        demo_terminal_navigation()

    if args.tool in ["all", "execute-cmd"]:
        demo_execute_command()

    if args.tool in ["all", "find-files"]:
        demo_find_files()

    if args.tool in ["all", "search-term"]:
        demo_search_term()

    if args.tool in ["all", "run-pytest"]:
        demo_run_pytest()


if __name__ == "__main__":
    main()
