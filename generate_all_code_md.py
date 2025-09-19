#!/usr/bin/env python3
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = root / "all_code.md"

    with output.open("w", encoding="utf-8") as md:
        # 统一收集并排序 .py / .yml / .yaml
        for p in sorted(
            [*root.rglob("*.py"), *root.rglob("*.yml"), *root.rglob("*.yaml")]
        ):
            relative_path = p.relative_to(root)
            md.write(f"{relative_path}\n")
            lang = "python" if p.suffix == ".py" else "yaml"
            md.write(f"'''{lang}\n")
            md.write(p.read_text(encoding="utf-8"))
            md.write("\n'''\n\n")


if __name__ == "__main__":
    main()
