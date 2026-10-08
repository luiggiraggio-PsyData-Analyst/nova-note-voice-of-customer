# NovaNote — Voice of the Customer (Exit Interview Analysis)

NovaNote is a hypothetical Notion-class productivity app used as a demonstration case.
This project analyzes why business accounts churn, end to end, with a transparent stack:
from interview corpus and sentiment analysis to an interactive theme network and report.

## What this project does

- Turns 10 exit-interview transcripts into a **coded, segmented corpus** (50 segments, 13 themes).
- Measures segment-level **sentiment** with a pretrained transformer (RoBERTa) cross-checked against
  a lexicon baseline (VADER) **and** hand-labeled gold data (accuracy / macro-F1 / Cohen's kappa).
- Models theme relationships as a **weighted co-occurrence network**: centrality, Louvain
  communities, correlation structure, propagation/influence indicators.
- Ships an **interactive dashboard** (themes & sentiment + explorable 3-D theme network)
  and this written report.

## Key findings

| Metric | Value |
|---|---|
| Interviews / segments / themes | 10 / 50 / 13 |
| Mean sentiment (all segments) | **-0.08** |
| Segments negative | **50%** |
| Most negative themes | support (-0.30), collaboration (-0.20), roadmap, integrations |
| Most praised | onboarding (+0.34), migration, competitor-comparison segments |
| Founders vs PMs mean sentiment | -0.06 vs +0.02 |
| RoBERTa (vs gold) | accuracy 0.60 · macro-F1 0.47 · kappa 0.36 |
| VADER (vs gold) | accuracy 0.40 · macro-F1 0.29 · kappa 0.15 |
| Model agreement (RoBERTa vs VADER) | kappa 0.67 |
| Most central theme (betweenness) | mobile |
| Communities (Louvain) | integrations+api / cost-and-trust / scale-collab-support |

**Top recommendation:** fix support (reply SLA + public bug roadmap) first, then pricing
transparency, performance at scale, collaboration permissions and API stability.

## Repository structure

```
data/raw/transcripts/   10 authored exit-interview transcripts (.txt)
data/raw/participants.csv
data/processed/         corpus_segments.csv, gold_labels.csv, themes_master.csv
scripts/
  01_build_corpus.py    segment + theme coding
  02_sentiment.py       RoBERTa + VADER + gold validation + aggregates/figures
  03_network.py         graph, centralities, communities, correlations, layout
  04_dashboard.py       Streamlit dashboard (2 views)
  05_report.py          generates docs/A1_report.pdf
outputs/                figures, tables, theme network (graphml/json)
docs/A1_report.pdf      the written report
.streamlit/config.toml  minimalist black/white/gray theme
```

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/01_build_corpus.py
python scripts/02_sentiment.py     # downloads ~550 MB RoBERTa model on first run
python scripts/03_network.py
python scripts/05_report.py        # -> docs/A1_report.pdf

streamlit run scripts/04_dashboard.py
```

## Stack (free / open-source)

Python 3.12 · pandas · numpy · transformers (RoBERTa) · NLTK/VADER · networkx ·
python-louvain · scipy/scikit-learn · matplotlib · wordcloud · Streamlit + Plotly ·
reportlab · Git/GitHub.

> The dataset is hypothetical and prepared for demonstration purposes. All code is clean
> of comments by design; the focus is on method transparency and business outcomes.