from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.data.load import read_text
from src.data.preprocess import normalize_text
from src.graphs.extract_events import extract_events, events_to_dict
from src.graphs.build_graph import build_event_graph, graph_to_jsonable

META_PATH = Path("data/metadata.csv")

LLM_DIR = Path("data/llm/same_culture")          # input retellings
OUT_ROOT = Path("outputs/llm_graphs/same_culture")  # output graphs


def read_title_and_body(path: Path) -> tuple[str, str]:
    """Assumes first non-empty line is title, rest is body."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    title = lines[i].strip() if i < len(lines) else path.stem
    body = "\n".join(lines[i + 1 :]).strip() if i + 1 < len(lines) else ""
    return title, body


def main() -> None:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Run your metadata builder first.")

    meta = pd.read_csv(META_PATH)
    if not {"story_id", "culture"}.issubset(meta.columns):
        raise ValueError("metadata.csv must contain at least: story_id, culture")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []

    for _, r in meta.iterrows():
        story_id = str(r["story_id"])
        culture = str(r["culture"])

        llm_path = LLM_DIR / f"{story_id}.txt"
        if not llm_path.exists():
            errors.append({"story_id": story_id, "culture": culture, "error": f"Missing LLM file: {llm_path}"})
            continue

        try:
            title, body = read_title_and_body(llm_path)
            if not body.strip():
                errors.append({"story_id": story_id, "culture": culture, "error": "Empty body after title line"})
                continue

            text = normalize_text(body)
            events = extract_events(text)
            G = build_event_graph(events)

            story_out = OUT_ROOT / story_id
            story_out.mkdir(parents=True, exist_ok=True)

            (story_out / "events.json").write_text(json.dumps(events_to_dict(events), indent=2), encoding="utf-8")
            (story_out / "graph.json").write_text(json.dumps(graph_to_jsonable(G), indent=2), encoding="utf-8")

            word_count = len(text.split())
            rows.append({
                "story_id": story_id,
                "culture": culture,
                "title": title,
                "word_count": word_count,
                "num_events": len(events),
                "events_per_100_words": (len(events) / word_count * 100) if word_count else 0.0,
                "input_path": str(llm_path.as_posix()),
                "output_dir": str(story_out.as_posix()),
            })

            print(f"[OK] {story_id} ({culture}) events={len(events)} words={word_count}")

        except Exception as e:
            errors.append({"story_id": story_id, "culture": culture, "error": repr(e)})
            print(f"[FAIL] {story_id}: {e}")

    summary_df = pd.DataFrame(rows).sort_values(["culture", "story_id"])
    summary_df.to_csv(OUT_ROOT / "summary.csv", index=False)

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(OUT_ROOT / "errors.csv", index=False)

    print("\nDone.")
    print(f"Saved summary: {OUT_ROOT / 'summary.csv'}")
    if len(errors_df) > 0:
        print(f"Some failed. See: {OUT_ROOT / 'errors.csv'}")
    else:
        print("No errors 🎉")


if __name__ == "__main__":
    main()