import re

import pandas as pd

from project_config import (
    PARTICIPANTS_CSV,
    PROCESSED,
    TRANSCRIPTS_DIR,
    THEMES,
    PATTERNS,
)

COMPILED = {t: re.compile(p, re.IGNORECASE) for t, p in PATTERNS.items()}


def clean_text(raw):
    text = re.sub(r"\s+", " ", raw).strip()
    return text


def split_segments(raw):
    return [seg for seg in re.split(r"\n\s*\n+", raw) if seg.strip()]


def assign_themes(text):
    lower = text.lower()
    found = [t for t in THEMES if COMPILED[t].search(lower)]
    return found


def main():
    participants = pd.read_csv(PARTICIPANTS_CSV)
    rows = []
    for _, p in participants.iterrows():
        fpath = TRANSCRIPTS_DIR / p["transcript_file"]
        raw = fpath.read_text(encoding="utf-8")
        for idx, seg in enumerate(split_segments(raw)):
            text = clean_text(seg)
            if not text:
                continue
            themes = assign_themes(text)
            row = {
                "file": p["transcript_file"],
                "seg_idx": idx,
                "participant_id": p["id"],
                "persona": p["persona"],
                "role": p["role"],
                "company_type": p["company_type"],
                "company_size": p["company_size"],
                "plan_tier": p["plan_tier"],
                "months_onboard": p["months_onboard"],
                "churn_reason": p["churn_reason"],
                "interview_date": p["interview_date"],
                "text": text,
                "word_count": len(text.split()),
                "n_themes": len(themes),
                "themes": ";".join(themes),
            }
            for t in THEMES:
                row[f"theme_{t}"] = 1 if t in themes else 0
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "corpus_segments.csv", index=False)
    master = pd.DataFrame({"theme": THEMES, "pattern": [PATTERNS[t] for t in THEMES]})
    master.to_csv(PROCESSED / "themes_master.csv", index=False)
    print(f"segments: {len(df)}")
    print("segments per file:")
    print(df.groupby("file").size().to_string())


if __name__ == "__main__":
    main()