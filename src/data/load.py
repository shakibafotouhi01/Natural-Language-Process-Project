from pathlib import Path

def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    path = Path(path)
    return path.read_text(encoding=encoding, errors="ignore")
