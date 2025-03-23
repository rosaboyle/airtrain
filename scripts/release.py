#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil


def run_command(command, error_message):
    """Run a shell command and handle errors"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n❌ {error_message}")
        print(f"Command output: {result.stdout}{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def bump_version():
    """Bump the version number"""
    subprocess.run(
        [sys.executable, os.path.join("scripts", "bump_version.py")], check=True
    )


def copy_changelog():
    """Copy changelog to the package directory"""
    shutil.copy("changelog.md", "airtrain/")


def clean_builds():
    """Clean previous build artifacts"""
    dirs_to_clean = ["dist", "build", "*.egg-info"]
    for dir_pattern in dirs_to_clean:
        run_command(f"rm -rf {dir_pattern}", f"Failed to clean {dir_pattern}")


def build_package():
    """Build the package"""
    run_command("python -m build", "Failed to build package")


def upload_to_pypi():
    """Upload to PyPI"""
    run_command("python -m twine upload dist/*", "Failed to upload to PyPI")


def main():
    try:
        # 1. Bump version
        print("🔼 Bumping version...")
        bump_version()

        # 2. Copy changelog
        print("📝 Copying changelog...")
        copy_changelog()

        # 3. Clean previous builds
        print("🧹 Cleaning previous builds...")
        clean_builds()

        # 4. Build package
        print("📦 Building package...")
        build_package()

        # 5. Upload to PyPI
        print("\n🚀 Uploading to PyPI...")
        upload_to_pypi()

        print("\n✨ Release completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during release process: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
