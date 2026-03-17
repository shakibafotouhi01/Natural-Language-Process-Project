from pathlib import Path
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------- Paths --------

META_PATH = Path("data/metadata.csv")

ORIG_ROOT = Path("outputs/human_graphs")
LLM_ROOT = Path("outputs/llm_graphs/same_culture")

OUT_PATH = Path("outputs/analysis_same_culture.csv")

SIM_THRESHOLD = 0.75


# -------- Model --------

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_events(path):
    data = json.loads(path.read_text())
    return [e["label"] for e in data]


def match_events(events_a, events_b):
    if not events_a or not events_b:
        return 0, 0, 0

    emb_a = model.encode(events_a)
    emb_b = model.encode(events_b)

    sim = cosine_similarity(emb_a, emb_b)

    matches = (sim.max(axis=1) >= SIM_THRESHOLD).sum()

    precision = matches / len(events_b)
    recall = matches / len(events_a)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


rows = []

meta = pd.read_csv(META_PATH)

for _, r in meta.iterrows():

    story_id = r["story_id"]
    culture = r["culture"]

    orig_events_path = ORIG_ROOT / story_id / "events.json"
    llm_events_path = LLM_ROOT / story_id / "events.json"

    if not orig_events_path.exists() or not llm_events_path.exists():
        continue

    events_orig = load_events(orig_events_path)
    events_llm = load_events(llm_events_path)

    p, r, f1 = match_events(events_orig, events_llm)

    rows.append({
        "story_id": story_id,
        "culture": culture,
        "precision": p,
        "recall": r,
        "f1": f1,
        "events_original": len(events_orig),
        "events_llm": len(events_llm)
    })

df = pd.DataFrame(rows)

df.to_csv(OUT_PATH, index=False)

print(df)
print("\nSaved to:", OUT_PATH)