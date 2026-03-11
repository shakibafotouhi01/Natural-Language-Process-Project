from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import spacy

# Load once (fast enough for small experiments)
_NLP = None

def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP

@dataclass(frozen=True)
class Event:
    sent_id: int
    subj: str
    verb: str
    obj: str
    sentence: str

    @property
    def label(self) -> str:
        # normalized label for graph nodes
        s = self.subj or "∅"
        o = self.obj or "∅"
        return f"{s} | {self.verb} | {o}"

def _find_subject(token) -> Optional[str]:
    # nsubj / nsubjpass
    for child in token.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child.lemma_.lower()
    return None

def _find_object(token) -> Optional[str]:
    # direct object / passive agent-ish / attribute / complement
    for child in token.children:
        if child.dep_ in ("dobj", "obj", "attr", "oprd"):
            return child.lemma_.lower()
    # prepositional object fallback (e.g., "talked to X")
    for child in token.children:
        if child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    return gc.lemma_.lower()
    return None

def extract_events(text: str) -> List[Event]:
    nlp = get_nlp()
    doc = nlp(text)

    events: List[Event] = []
    sent_id = 0

    for sent in doc.sents:
        # choose the ROOT verb-like token for the sentence
        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break
        if root is None:
            sent_id += 1
            continue

        # ignore very non-verb roots (rare but happens)
        if root.pos_ not in ("VERB", "AUX"):
            sent_id += 1
            continue

        subj = _find_subject(root) or ""
        obj = _find_object(root) or ""
        verb = root.lemma_.lower()

        # basic filter: skip extremely empty events
        if not verb:
            sent_id += 1
            continue

        events.append(Event(
            sent_id=sent_id,
            subj=subj,
            verb=verb,
            obj=obj,
            sentence=sent.text.strip()
        ))
        sent_id += 1

    return events

def events_to_dict(events: List[Event]) -> List[Dict[str, Any]]:
    return [
        {
            "sent_id": e.sent_id,
            "subj": e.subj,
            "verb": e.verb,
            "obj": e.obj,
            "label": e.label,
            "sentence": e.sentence,
        }
        for e in events
    ]
