from pathlib import Path
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL = "gpt-4.1-mini"

META_PATH = Path("data/metadata.csv")
OUT_DIR = Path("data/llm/cross_culture")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are a careful literary reteller adapting stories across cultures.
Always write the output in English only.
Do not switch languages.
Preserve the main sequence of events, main characters, and causal structure.
Adapt setting, imagery, symbols, and cultural references to the target culture.
Keep the retelling close in length to the original.
Output only the retold story text.
""".strip()


def normalize_culture_name(culture: str) -> str:
    return culture.strip().lower().replace("_", " ")


culture_map = {
    "german": "Persian",
    "persian": "Native American",
    "native american": "German",
}


def read_story(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # first non-empty line = title
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    title = lines[i].strip() if i < len(lines) else path.stem
    body = "\n".join(lines[i + 1:]).strip() if i + 1 < len(lines) else ""

    return title, body


meta = pd.read_csv(META_PATH)

for _, row in meta.iterrows():
    story_id = str(row["story_id"])
    culture_raw = str(row["culture"])
    culture = normalize_culture_name(culture_raw)

    if culture not in culture_map:
        print(f"[SKIP] Unknown culture value: {culture_raw!r}")
        continue

    target_culture = culture_map[culture]
    path = Path(row["file_path"])

    if not path.exists():
        print(f"[SKIP] Missing file: {path}")
        continue

    title, body = read_story(path)

    if not body.strip():
        print(f"[SKIP] Empty body: {story_id}")
        continue

    prompt = f"""Retell the following story as if it originated in {target_culture} folklore.

Important constraints:
- Preserve the main sequence of events.
- Preserve the main characters and causal structure.
- Write the entire retelling in English only.
- Adapt the setting, imagery, symbols, and cultural references so the story feels natural in {target_culture} folklore.
- Keep the retelling roughly the same length as the original (±20%).
- Output only the retold story text.

Story title: {title}

Story:
{body}
""".strip()

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.output_text.strip()
        out_path = OUT_DIR / f"{story_id}.txt"
        out_path.write_text(f"{title}\n\n{text}\n", encoding="utf-8")

        print(f"[OK] Generated: {story_id} | {culture_raw} -> {target_culture}")
        time.sleep(0.5)

    except Exception as e:
        print(f"[FAIL] {story_id}: {e}")
