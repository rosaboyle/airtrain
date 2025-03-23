#!/usr/bin/env python3
"""
Release script for TRMX
This script handles the entire release process:
1. Bumping the version number (patch, minor, or major)
2. Building the package
3. Uploading to PyPI
"""

import os
import sys
import re
import toml
import subprocess
import argparse


def run_command(command, error_message):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n❌ {error_message}")
        print(f"Command output: {result.stdout}{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def read_version():
    """Read current version from __init__.py"""
    init_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "trmx_agent", "__init__.py"
    )
    with open(init_path, "r", encoding="utf-8") as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def bump_version(version_str, release_type="patch"):
    """Increment the version according to release type (patch, minor, or major)"""
    major, minor, patch = map(int, version_str.split("."))
    
    if release_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif release_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif release_type == "major":
        return f"{major + 1}.0.0"
    else:
        raise ValueError(f"Unknown release type: {release_type}")


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


def update_version(release_type="patch"):
    """Update version in all configuration files"""
    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(__file__))

    # Get the current version and bump it
    current_version = read_version()
    new_version = bump_version(current_version, release_type)

    # Update version in __init__.py
    init_path = os.path.join(root_dir, "trmx_agent", "__init__.py")
    update_init_version(new_version, init_path)

    # Update version in pyproject.toml
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    update_pyproject_version(new_version, pyproject_path)

    print(f"✅ Updated version to {new_version} in all configuration files")
    return new_version


def clean_builds():
    """Clean previous build artifacts"""
    dirs_to_clean = ["dist", "build", "*.egg-info"]
    for dir_pattern in dirs_to_clean:
        run_command(f"rm -rf {dir_pattern}", f"Failed to clean {dir_pattern}")
    print("✅ Cleaned previous build artifacts")


def build_package():
    """Build the package"""
    run_command("python -m pip install --upgrade build", "Failed to install build")
    run_command("python -m build", "Failed to build package")
    print("✅ Built package")


def upload_to_pypi():
    """Upload to PyPI"""
    run_command("python -m pip install --upgrade twine", "Failed to install twine")
    run_command("python -m twine upload dist/*", "Failed to upload to PyPI")
    print("✅ Uploaded to PyPI")


def main():
    parser = argparse.ArgumentParser(description='Release TRMX package to PyPI')
    parser.add_argument('--type', choices=['patch', 'minor', 'major'], 
                        default='patch', help='Type of release (patch, minor, major)')
    parser.add_argument('--skip-upload', action='store_true', 
                        help='Skip uploading to PyPI (build only)')
    args = parser.parse_args()

    try:
        # 1. Bump version
        print(f"\n🔼 Bumping {args.type} version...")
        new_version = update_version(args.type)

        # 2. Clean previous builds
        print("\n🧹 Cleaning previous builds...")
        clean_builds()

        # 3. Build package
        print("\n📦 Building package...")
        build_package()

        if not args.skip_upload:
            # 4. Upload to PyPI
            print("\n🚀 Uploading to PyPI...")
            upload_to_pypi()

        print(f"\n✨ Release {new_version} completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during release process: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 