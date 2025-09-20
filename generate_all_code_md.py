#!/usr/bin/env python3
from pathlib import Path


def is_bash_script(p: Path) -> bool:
    """只读首行，判断 shebang 是否指向 bash。"""
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        return first.startswith("#!") and "bash" in first
    except Exception:
        return False


def main():
    root = Path(__file__).resolve().parent
    output = root / "all_code.md"

    # 收集：.py/.yml/.yaml + .sh/.bash +（无扩展名但 shebang 指向 bash）
    candidates = set()
    for pat in ("*.py", "*.yml", "*.yaml", "*.sh", "*.bash"):
        candidates.update(root.rglob(pat))
    for p in root.rglob("*"):
        if p.is_file() and not p.suffix and is_bash_script(p):
            candidates.add(p)

    with output.open("w", encoding="utf-8") as md:
        for p in sorted(candidates):
            relative_path = p.relative_to(root)
            md.write(f"{relative_path}\n")
            if p.suffix == ".py":
                lang = "python"
            elif p.suffix in {".yml", ".yaml"}:
                lang = "yaml"
            elif p.suffix in {".sh", ".bash"} or (not p.suffix and is_bash_script(p)):
                lang = "bash"
            else:
                continue  # 理论上不会走到这里；保守起见跳过
            md.write(f"'''{lang}\n")
            md.write(p.read_text(encoding="utf-8"))
            md.write("\n'''\n\n")


if __name__ == "__main__":
    main()
