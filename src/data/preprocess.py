import re

def normalize_text(text: str) -> str:
    # basic cleanup: whitespace + remove very noisy Gutenberg-like headers if present
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # For copies from Gutenberg, this helps:
    # remove anything before/after common markers (safe if markers not found)
    start_markers = ["*** START OF", "***START OF", "START OF THIS PROJECT GUTENBERG EBOOK"]
    end_markers = ["*** END OF", "***END OF", "END OF THIS PROJECT GUTENBERG EBOOK"]

    lower = text.lower()
    for m in start_markers:
        i = lower.find(m.lower())
        if i != -1:
            text = text[i:]
            break

    lower = text.lower()
    for m in end_markers:
        i = lower.find(m.lower())
        if i != -1:
            text = text[:i]
            break

    return text.strip()
