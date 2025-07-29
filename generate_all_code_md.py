#!/usr/bin/env python3
"""
Generate a markdown file aggregating all Python source code in the project.
Each entry begins with the file path followed by the code enclosed in
triple single quotes marked as a Python block.
"""

from pathlib import Path


def main():
    """Search for Python files and assemble them into all_code.md."""
    root = Path(__file__).resolve().parent  # Project root assumed to be script location
    output = root / "all_code.md"

    with output.open("w", encoding="utf-8") as md:
        for py in sorted(root.rglob("*.py")):
            relative_path = py.relative_to(root)
            md.write(f"{relative_path}\n")
            md.write("'''python\n")
            md.write(py.read_text(encoding="utf-8"))
            md.write("\n'''\n\n")


if __name__ == "__main__":
    main()

