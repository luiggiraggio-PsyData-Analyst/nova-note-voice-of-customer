import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from project_config import PROCESSED, FIGURES, TABLES, NETWORK, THEMES, ACTION_BY_CLUSTER

st.set_page_config(page_title="NovaNote VoC", layout="wide", page_icon="")

FONT = "monospace"

st.markdown(
    """
<style>
html, body, [class*="css"], .stApp { font-family: 'DejaVu Sans Mono','Ubuntu Mono','Consolas',monospace !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; }
.stMetric { background: #f2f2f2; border-radius: 0; padding: 0.6rem 0.8rem; }
[data-testid="stSidebar"] { background: #fafafa; }
h1, h2, h3 { letter-spacing: 0.02em; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_segments():
    return pd.read_csv(TABLES / "segment_sentiment.csv")


@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(TABLES / "theme_metrics.csv")


@st.cache_data(show_spinner=False)
def load_edges():
    return pd.read_csv(TABLES / "theme_edges.csv")


@st.cache_data(show_spinner=False)
def load_corr():
    return pd.read_csv(TABLES / "theme_correlation.csv", index_col=0)


@st.cache_data(show_spinner=False)
def load_layout():
    return json.loads((NETWORK / "layout3d.json").read_text())


@st.cache_data(show_spinner=False)
def load_senti_theme():
    return pd.read_csv(TABLES / "sentiment_by_theme.csv")


@st.cache_data(show_spinner=False)
def load_senti_persona():
    return pd.read_csv(TABLES / "sentiment_by_persona.csv")


@st.cache_data(show_spinner=False)
def long_theme(segs):
    melt = segs.melt(
        id_vars=["file", "seg_idx", "persona", "interview_date", "text", "score", "label"],
        value_vars=[f"theme_{t}" for t in THEMES],
        var_name="k", value_name="present",
    )
    melt = melt[melt["present"] == 1]
    melt["theme"] = melt["k"].str.replace("theme_", "")
    melt["month"] = pd.to_datetime(melt["interview_date"]).dt.to_period("M").astype(str)
    return melt


@st.cache_data(show_spinner=False)
def wordcloud_image(segs):
    text = " ".join(segs["text"].astype(str))
    wc = WordCloud(
        width=800, height=360, background_color="white", colormap="Greys",
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", max_words=120,
    ).generate(text)
    return wc.to_image()


def shades(n):
    return ["#111111", "#4a4a4a", "#848484", "#a9a9a9", "#c6c6c6"][:max(1, n)]


def net_fig(edges, layout, metrics, communities, size_metric, highlight=None, labels=True):
    X = {t: layout[t][0] for t in THEMES}
    Y = {t: layout[t][1] for t in THEMES}
    Z = {t: layout[t][2] for t in THEMES}
    node_size = metrics[size_metric].replace(0, np.nan)
    node_size = 14 + 42 * (node_size - node_size.min()) / (node_size.max() - node_size.min() + 1e-9)
    node_size = node_size.fillna(14)

    cmap = dict(zip(sorted(metrics["community"].unique()), shades(metrics["community"].nunique())))
    node_color = [cmap[c] for c in metrics["community"]]

    fig = go.Figure()
    x, y, z = [], [], []
    for _, r in edges.iterrows():
        if highlight and highlight not in (r["source"], r["target"]):
            continue
        for t in (r["source"], r["target"]):
            x.append(X[t]); y.append(Y[t]); z.append(Z[t])
            x.append(None); y.append(None); z.append(None)
    if x:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="#b0b0b0", width=1.4), hoverinfo="none", showlegend=False,
        ))
    hover = [
        f"{t}<br>degree {m.degree}<br>betweenness {m.betweenness}<br>pagerank {m.pagerank}<br>community {m.community}"
        for t, m in metrics.set_index("theme").iterrows()
    ]
    fig.add_trace(go.Scatter3d(
        x=[X[t] for t in THEMES], y=[Y[t] for t in THEMES], z=[Z[t] for t in THEMES],
        mode="markers", marker=dict(size=list(node_size), color=node_color,
                                    line=dict(color="#111111", width=1)),
        text=hover, hoverinfo="text", showlegend=False,
    ))
    if labels:
        fig.add_trace(go.Scatter3d(
            x=[X[t] for t in THEMES], y=[Y[t] for t in THEMES], z=[Z[t] for t in THEMES],
            mode="text", text=THEMES, textfont=dict(color="#111111", size=11, family="monospace"),
            hoverinfo="none", showlegend=False,
        ))
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff", margin=dict(l=0, r=0, t=0, b=0),
        uirevision="fixed", scene=dict(
            xaxis=dict(showgrid=True, zeroline=False, showbackground=False, gridcolor="#e6e6e6"),
            yaxis=dict(showgrid=True, zeroline=False, showbackground=False, gridcolor="#e6e6e6"),
            zaxis=dict(showgrid=True, zeroline=False, showbackground=False, gridcolor="#e6e6e6"),
            dragmode="orbit", aspectmode="cube",
        ),
    )
    return fig


def page_themes(segs, melt, metrics):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exit interviews", segs["participant_id"].nunique())
    c2.metric("Segments", len(segs))
    c3.metric("Mean sentiment", round(segs["score"].mean(), 2))
    c4.metric("Positive %", f"{100 * (segs['label'] == 'positive').mean():.0f}")
    c5.metric("Negative %", f"{100 * (segs['label'] == 'negative').mean():.0f}")

    st.markdown("#### Theme distribution")
    freq = melt["theme"].value_counts().reindex(THEMES).fillna(0)
    f1 = go.Figure(go.Bar(
        x=freq.values, y=freq.index, orientation="h",
        marker_color=["#111111" if v >= freq.mean() else "#d9d9d9" for v in freq.values],
    ))
    f1.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=8, b=0), height=360,
                     xaxis_title="segment mentions", font=dict(family=FONT))
    st.plotly_chart(f1, use_container_width=True)

    st.markdown("#### Sentiment by theme")
    senti = load_senti_theme().set_index("theme").reindex(THEMES)
    f2 = go.Figure(go.Bar(
        x=senti["mean_score"], y=senti.index, orientation="h",
        marker_color=["#111111" if v <= 0 else "#c8c8c8" for v in senti["mean_score"]],
    ))
    f2.add_vline(x=0, line_color="#111111", line_width=1)
    f2.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=8, b=0), height=380,
                     xaxis_title="mean sentiment (-1 .. +1)", font=dict(family=FONT))
    st.plotly_chart(f2, use_container_width=True)

    st.markdown("#### Theme x sentiment shares")
    matrix = pd.crosstab(melt["theme"], melt["label"], normalize="index").reindex(THEMES).fillna(0)
    for col in ("positive", "neutral", "negative"):
        if col not in matrix.columns:
            matrix[col] = 0.0
    f3 = go.Figure(go.Heatmap(
        z=matrix[["negative", "neutral", "positive"]].values,
        x=["negative", "neutral", "positive"], y=matrix.index, colorscale="Greys", zmin=0, zmax=1,
    ))
    f3.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=8, b=0), height=380,
                     font=dict(family=FONT))
    st.plotly_chart(f3, use_container_width=True)

    st.markdown("#### Sentiment timeline")
    segm = segs.copy()
    segm["month"] = pd.to_datetime(segm["interview_date"]).dt.to_period("M").astype(str)
    ts = segm.groupby("month")["score"].mean().reset_index()
    f4 = go.Figure(go.Scatter(x=ts["month"], y=ts["score"], mode="lines+markers",
                              line=dict(color="#111111", width=2)))
    f4.add_hline(y=0, line_dash="dash", line_color="#9a9a9a")
    f4.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=8, b=0), height=300,
                     xaxis_title="month", font=dict(family=FONT))
    st.plotly_chart(f4, use_container_width=True)

    st.markdown("#### Persona comparison")
    persona = load_senti_persona().set_index("persona").reindex(["Founder", "Product Manager"])
    f5 = go.Figure(go.Bar(
        x=persona.index, y=persona["score"],
        marker_color=["#111111", "#8f8f8f"],
    ))
    f5.add_hline(y=0, line_color="#111111", line_width=1)
    f5.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=8, b=0), height=300,
                     yaxis_title="mean sentiment", font=dict(family=FONT))
    st.plotly_chart(f5, use_container_width=True)

    st.markdown("#### Word cloud")
    st.image(wordcloud_image(segs), use_container_width=True)

    st.markdown("#### Churn-driver ranking")
    ranked = load_senti_theme().assign(driver_rank=range(1, len(load_senti_theme()) + 1))
    ranked = ranked[["theme", "mean_score", "n_segments", "negative_share", "driver_rank"]]
    st.dataframe(ranked.sort_values("mean_score"), use_container_width=True, hide_index=True)

    st.markdown("#### Quotes explorer")
    colf1, colf2, colf3, colf4 = st.columns(4)
    tsel = colf1.multiselect("Theme", THEMES, default=[])
    ssel = colf2.selectbox("Sentiment", ["all"] + ["negative", "neutral", "positive"])
    psel = colf3.multiselect("Persona", ["Founder", "Product Manager"], default=["Founder", "Product Manager"])
    q = colf4.text_input("Keyword")
    view = segs.copy()
    if tsel:
        keep = view[[f"theme_{t}" for t in tsel]].any(axis=1)
        view = view[keep]
    if ssel != "all":
        view = view[view["label"] == ssel]
    view = view[view["persona"].isin(psel)]
    if q:
        view = view[view["text"].str.contains(q, case=False, na=False)]
    st.dataframe(
        view[["file", "persona", "label", "score", "text"]].sort_values("score"),
        use_container_width=True, hide_index=True,
    )


def page_network(metrics, edges, layout):
    metric_opts = ["degree", "weighted_degree", "betweenness", "closeness", "eigenvector", "pagerank"]
    colc1, colc2, colc3 = st.columns([1, 2, 1])
    size_metric = colc1.selectbox("Node size metric", metric_opts, index=2)
    highlight = colc2.selectbox("Highlight theme", ["none"] + THEMES)
    show_labels = colc3.checkbox("Show labels", value=True)
    hl = None if highlight == "none" else highlight

    st.plotly_chart(
        net_fig(edges, layout, metrics, None, size_metric, hl, show_labels),
        use_container_width=True,
    )

    st.markdown("#### Theme correlation matrix")
    corr = load_corr()
    cf = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, colorscale="Greys", zmin=-1, zmax=1,
    ))
    cf.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=8, b=0), height=460,
                     font=dict(family=FONT, size=10))
    st.plotly_chart(cf, use_container_width=True)

    st.markdown("#### Top by selected metric")
    top = metrics.nlargest(10, size_metric)[["theme", size_metric, "community"]]
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("#### Community -> recommended action")
    rows = []
    cdf = pd.read_csv(TABLES / "communities.csv")
    for _, r in cdf.iterrows():
        themes_in = [t.strip() for t in r["themes"].split(",")]
        key = None
        for cand, msg in ACTION_BY_CLUSTER.items():
            priors = {
                "trust_cost": ["pricing", "onboarding", "migration"],
                "scale_code": ["api", "integrations"],
                "collab_structure": ["collaboration", "templates"],
                "support_ops": ["support", "roadmap"],
                "mobile_security": ["mobile", "security"],
            }
            if any(t in themes_in for t in priors[cand]):
                key, msg_local = cand, msg
                break
        rows.append({"community": r["community"], "themes": r["themes"], "action": msg_local if key else msg})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


segs = load_segments()
melt = long_theme(segs)
metrics = load_metrics()
edges = load_edges()
layout = load_layout()

page = st.sidebar.radio("View", ["Themes & Sentiment", "Theme Network"])
st.sidebar.markdown("**NovaNote — exit-interview study**")
st.sidebar.caption("50 segments x 13 themes | RoBERTa sentiment | NetworkX")

if page == "Themes & Sentiment":
    page_themes(segs, melt, metrics)
else:
    page_network(metrics, edges, layout)