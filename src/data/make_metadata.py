from pathlib import Path
import csv
import re

# Note: The web sources were included later manually 

DATA_ROOT = Path("data/human")          # culture folders live here
OUT_PATH = Path("data/metadata.csv")    # output CSV

def slugify(text: str) -> str:
    """Turn a filename into a stable ID-ish string."""
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:60] if text else "story"

rows = []
for culture_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
    culture = culture_dir.name

    for txt_path in sorted(culture_dir.glob("*.txt")):
        # story_id = filename without extension 
        story_id = txt_path.stem

        # title = first non-empty line of the file
        title = ""
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    title = line
                    break

        if not title:
            # fallback: derive title from filename if file starts with blanks
            title = txt_path.stem.replace("_", " ").strip().title()

        rows.append({
            "story_id": story_id,
            "culture": culture,
            "title": title,
            "file_path": str(txt_path.as_posix()),
        })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["story_id", "culture", "title", "file_path"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_PATH}")
print("Sample:", rows[:3])
