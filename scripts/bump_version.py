import os
import re
import toml
import sys


def read_version():
    """Read current version from __init__.py"""
    init_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "airtrain", "__init__.py"
    )
    with open(init_path, "r", encoding="utf-8") as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def bump_patch_version(version_str):
    """Increment the patch version"""
    major, minor, patch = map(int, version_str.split("."))
    return f"{major}.{minor}.{patch + 1}"


def update_init_version(new_version, init_path):
    """Update version in __init__.py"""
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        f'__version__ = "{new_version}"',
        content,
        flags=re.M,
    )

    with open(init_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def update_pyproject_version(new_version, pyproject_path):
    """Update version in pyproject.toml"""
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = toml.load(f)

    content["project"]["version"] = new_version

    with open(pyproject_path, "w", encoding="utf-8") as f:
        toml.dump(content, f)


def update_version(new_version=None):
    """Update version in all configuration files"""
    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(__file__))

    # If no version provided, bump the current version
    if new_version is None:
        current_version = read_version()
        new_version = bump_patch_version(current_version)

    # Update version in __init__.py
    init_path = os.path.join(root_dir, "airtrain", "__init__.py")
    update_init_version(new_version, init_path)

    # Update version in pyproject.toml
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    update_pyproject_version(new_version, pyproject_path)

    print(f"✅ Updated version to {new_version} in all configuration files")
    return new_version


def main():
    if len(sys.argv) > 1:
        new_version = sys.argv[1]
    else:
        new_version = None
    return update_version(new_version)


if __name__ == "__main__":
    main()
