from __future__ import annotations

import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -------- Config --------
MODEL = "gpt-4.1-mini"  # good balance of quality/speed for rewriting tasks :contentReference[oaicite:1]{index=1}
SLEEP_SECONDS = 0.6     # gentle pacing to avoid rate-limit pain

META_PATH = Path("data/metadata.csv")
HUMAN_ROOT = Path("data/human")
OUT_DIR = Path("data/llm/same_culture")

# -------- Prompting --------
SYSTEM_PROMPT = (
    "You are a careful literary reteller. "
    "Retell stories in modern English while preserving characters and the sequence of key events. "
    "Do not add commentary, explanations, headings, or bullet points. Output only the retold story."
)

def build_user_prompt(culture: str, title: str, story_body: str) -> str:
    return f"""
Task:
Retell the story below in modern English while preserving the main characters and the sequence of key events.
Keep the cultural setting and narrative flavor consistent with its origin: {culture}.
Keep length roughly similar to the original (±20%).
Avoid summarizing: keep the events.

Output format:
- Output ONLY the retold story text (no analysis).
- Do NOT include the original story.

Story title: {title}

Story:
{story_body}
""".strip()

# -------- Helpers --------
def read_story(path: Path) -> tuple[str, str]:
    """
    Returns (title, body). Assumes first non-empty line is title.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # find title
    title = ""
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        title = lines[i].strip()
        i += 1
    else:
        title = path.stem

    # body = rest (keep paragraph breaks)
    body = "\n".join(lines[i:]).strip()
    return title, body

def write_output(out_path: Path, title: str, retelling: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep first line as title for consistency with human files
    out_path.write_text(f"{title}\n\n{retelling.strip()}\n", encoding="utf-8")

def main() -> None:
    load_dotenv() 
    client = OpenAI()

    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Create it first.")

    meta = pd.read_csv(META_PATH)
    if not {"story_id", "culture", "file_path"}.issubset(meta.columns):
        raise ValueError("metadata.csv must contain columns: story_id, culture, file_path")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    for _, row in meta.iterrows():
        story_id = str(row["story_id"])
        culture = str(row["culture"])
        file_path = Path(str(row["file_path"]))

        if not file_path.exists():
            print(f"[SKIP] Missing file: {file_path}")
            continue

        out_path = OUT_DIR / f"{story_id}.txt"
        if out_path.exists():
            print(f"[SKIP] Already exists: {out_path}")
            continue

        title, body = read_story(file_path)
        if not body:
            print(f"[SKIP] Empty story body: {file_path}")
            continue

        user_prompt = build_user_prompt(culture=culture, title=title, story_body=body)

        # Responses API call (official reference) :contentReference[oaicite:3]{index=3}
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        retelling_text = resp.output_text.strip()
        write_output(out_path, title=title, retelling=retelling_text)

        done += 1
        print(f"[OK] {story_id} ({culture}) -> {out_path}")

        time.sleep(SLEEP_SECONDS)

    print(f"Done. Generated {done} same-culture retellings into {OUT_DIR}")

if __name__ == "__main__":
    main()
