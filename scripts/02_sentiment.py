import math

import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from project_config import (
    MODEL_NAME,
    PROCESSED,
    FIGURES,
    TABLES,
    THEMES,
    SENTIMENT_LABELS,
    VADER_POS,
    VADER_NEG,
)

GRAY = ["#111111", "#3a3a3a", "#767676", "#b0b0b0", "#e0e0e0"]
POS = "#444444"
NEU = "#9a9a9a"
NEG = "#d0d0d0"

plt.rcParams.update({
    "font.family": "DejaVu Sans Mono",
    "axes.edgecolor": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "text.color": "#111111",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def classify_batch(texts, tokenizer, model, bs=16):
    scores = []
    labels = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            for row in probs:
                scores.append(float(row[2] - row[0]))
                labels.append(SENTIMENT_LABELS[int(np.argmax(row))])
    return np.asarray(scores), np.asarray(labels)


def vader_scores(texts):
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    return np.asarray([analyzer.polarity_scores(t)["compound"] for t in texts])


def score_to_label(score):
    if score > VADER_POS:
        return "positive"
    if score < VADER_NEG:
        return "negative"
    return "neutral"


def classify_vader(compounds):
    return [score_to_label(c) for c in compounds]


def make_figure(df):
    agg = (
        df.groupby("theme")
        .agg(mean_score=("score", "mean"), n=("text", "count"))
        .sort_values("mean_score")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#111111" if v <= -0.2 else "#9a9a9a" if v >= 0.2 else "#d5d5d5" for v in agg["mean_score"]]
    ax.barh(agg["theme"], agg["mean_score"], color=colors, edgecolor="#111111", linewidth=0.6)
    ax.axvline(0, color="#111111", linewidth=0.8)
    ax.set_xlabel("mean sentiment score (-1 negative / +1 positive)")
    ax.set_title("mean sentiment by theme")
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sentiment_by_theme.png", dpi=180)
    plt.close(fig)
    return agg


def persona_figure(df):
    p = df.groupby(["persona"])["score"].mean().reindex(["Founder", "Product Manager"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = ["#111111", "#8f8f8f"]
    ax.bar(p["persona"], p["score"], color=cols, edgecolor="#111111", linewidth=0.6, width=0.5)
    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.set_ylabel("mean sentiment score")
    ax.set_title("mean sentiment by persona")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sentiment_by_persona.png", dpi=180)
    plt.close(fig)


def timeline_figure(df):
    df["month"] = pd.to_datetime(df["interview_date"]).dt.to_period("M").astype(str)
    ts = df.groupby("month")["score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ts["month"], ts["score"], color="#111111", linewidth=1.6, marker="o", markersize=4)
    ax.axhline(0, color="#9a9a9a", linewidth=0.8, linestyle="--")
    ax.set_xlabel("month")
    ax.set_ylabel("mean sentiment score")
    ax.set_title("mean sentiment by month")
    ax.tick_params(axis="x", rotation=30)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sentiment_timeline.png", dpi=180)
    plt.close(fig)


def heatmap_figure(df):
    matrix = pd.crosstab(df["theme"], df["label"], normalize="index")
    for col in ("positive", "neutral", "negative"):
        if col not in matrix.columns:
            matrix[col] = 0.0
    matrix = matrix[["positive", "neutral", "negative"]]
    fig, ax = plt.subplots(figsize=(5.4, len(matrix) * 0.42 + 1))
    im = ax.imshow(matrix.values, cmap="Greys", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["positive", "neutral", "negative"])
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("theme x sentiment share")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center",
                    color="#111111", fontsize=7)
    fig.colorbar(im, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "theme_sentiment_heatmap.png", dpi=180)
    plt.close(fig)
    return matrix


def main():
    segs = pd.read_csv(PROCESSED / "corpus_segments.csv")
    texts = segs["text"].tolist()
    tokenizer, model = load_model()

    roberta, roberta_labels = classify_batch(texts, tokenizer, model, bs=16)
    vader = vader_scores(texts)

    segs["score"] = np.clip(roberta, -1, 1)
    segs["vader_compound"] = vader
    segs["label"] = roberta_labels
    segs["vader_label"] = classify_vader(vader)

    segs.to_csv(TABLES / "segment_sentiment.csv", index=False)

    gold = pd.read_csv(PROCESSED / "gold_labels.csv").rename(columns={"transcript_file": "file"})
    merged = segs.merge(gold, on=["file", "seg_idx"], how="inner")

    reliability = []
    for name, pred in (("roberta", merged["label"]), ("vader", merged["vader_label"])):
        gold_lab = merged["gold_label"]
        acc = accuracy_score(gold_lab, pred)
        f1 = f1_score(gold_lab, pred, average="macro")
        kappa = cohen_kappa_score(gold_lab, pred)
        reliability.append(
            {"model": name, "accuracy": round(float(acc), 3), "macro_f1": round(float(f1), 3),
             "cohen_kappa": round(float(kappa), 3)}
        )
    reliability.append(
        {"model": "roberta_vader", "accuracy": None, "macro_f1": None,
         "cohen_kappa": round(float(np.corrcoef(merged["score"], merged["vader_compound"])[0, 1]), 3)}
    )
    rel = pd.DataFrame(reliability)
    rel.to_csv(TABLES / "reliability_metrics.csv", index=False)
    print(rel.to_string(index=False))

    cm = confusion_matrix(merged["gold_label"], merged["label"], labels=SENTIMENT_LABELS)
    cmdf = pd.DataFrame(cm, index=["gold:" + x for x in SENTIMENT_LABELS], columns=["pred:" + x for x in SENTIMENT_LABELS])
    cmdf.to_csv(TABLES / "roberta_confusion_matrix.csv")

    long = segs.melt(id_vars=["file", "seg_idx", "text", "persona", "interview_date", "score", "label"],
                     value_vars=[f"theme_{t}" for t in THEMES],
                     var_name="theme_flag", value_name="present")
    long = long[long["present"] == 1]
    long["theme"] = long["theme_flag"].str.replace("theme_", "")
    theme_sent = long.groupby("theme").agg(
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        n_segments=("text", "count"),
        n_negative=("label", lambda s: (s == "negative").sum()),
        n_positive=("label", lambda s: (s == "positive").sum()),
    ).reset_index().sort_values("mean_score")
    theme_sent["negative_share"] = (theme_sent["n_negative"] / theme_sent["n_segments"]).round(3)
    theme_sent.to_csv(TABLES / "sentiment_by_theme.csv", index=False)

    persona_sent = long.groupby("persona")["score"].mean().reset_index()
    persona_sent.to_csv(TABLES / "sentiment_by_persona.csv", index=False)

    segs["month"] = pd.to_datetime(segs["interview_date"]).dt.to_period("M").astype(str)
    month_sent = segs.groupby("month")["score"].mean().reset_index()
    month_sent.to_csv(TABLES / "sentiment_by_month.csv", index=False)

    make_figure(long)
    persona_figure(segs)
    timeline_figure(segs)
    heatmap_figure(long)

    print("wrote tables + figures ok")


if __name__ == "__main__":
    main()