import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable, SimpleDocTemplate,
)

from project_config import FIGURES, TABLES, DOCS

M = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("Mono", f"{M}/DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("Mono-B", f"{M}/DejaVuSansMono-Bold.ttf"))

BLACK = colors.HexColor("#111111")
GREY = colors.HexColor("#555555")
LGREY = colors.HexColor("#e9e9e9")

body = ParagraphStyle("body", fontName="Mono", fontSize=8.6, leading=12.4,
                      alignment=TA_JUSTIFY, textColor=BLACK, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Mono-B", fontSize=15, leading=18, textColor=BLACK,
                    spaceBefore=14, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Mono-B", fontSize=11, leading=14, textColor=GREY,
                    spaceBefore=10, spaceAfter=5)
sub = ParagraphStyle("sub", fontName="Mono", fontSize=9.5, leading=13, textColor=GREY,
                     spaceAfter=10)
cell = ParagraphStyle("cell", fontName="Mono", fontSize=7.6, leading=10, textColor=BLACK)
cellh = ParagraphStyle("cellh", fontName="Mono-B", fontSize=7.8, leading=10, textColor=colors.white)


def P(text, style=body):
    return Paragraph(text, style)


def table(df, widths=None):
    data = [[Paragraph(str(c), cellh) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(str(v), cell) for v in r.values])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLACK),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def img(fname, width=16.5 * cm):
    return Image(str(FIGURES / fname), width=width, height=width * 0.72)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Mono", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#d0d0d0"))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()


ACTIONS = {
    0: "API & integrations cluster - stabilize the public API (SDK, rate limits), ship native "
       "connectors and publish an integration roadmap.",
    1: "Cost & trust cluster - transparent pricing with no surprise billing, clearer plan "
       "feature tables, official import/export, and published security/residency + SSO docs.",
    2: "Scale & collaboration cluster - support reply SLA and public known-bug status, "
       "document-level permissions, assignable comments, versionable locked templates, "
       "offline-first mobile, and performance work at enterprise scale.",
}


def build():
    story = []
    segs = pd.read_csv(TABLES / "segment_sentiment.csv")
    rel = pd.read_csv(TABLES / "reliability_metrics.csv")
    cm_ = pd.read_csv(TABLES / "roberta_confusion_matrix.csv", index_col=0)
    senti = pd.read_csv(TABLES / "sentiment_by_theme.csv")
    persona = pd.read_csv(TABLES / "sentiment_by_persona.csv")
    month = pd.read_csv(TABLES / "sentiment_by_month.csv")
    metrics = pd.read_csv(TABLES / "theme_metrics.csv")
    communities = pd.read_csv(TABLES / "communities.csv")

    n_seg = len(segs)
    mean_score = round(segs["score"].mean(), 2)
    neg_pct = round(100 * (segs["label"] == "negative").mean())
    least = senti.sort_values("mean_score").head(3)
    best = senti.sort_values("mean_score").tail(2)
    founder = round(persona.set_index("persona")["score"]["Founder"], 2)
    pm = round(persona.set_index("persona")["score"]["Product Manager"], 2)

    execsum = (
        f"NovaNote, a Notion-class productivity suite, is losing business accounts. This study analyzes "
        f"10 exit interviews (50 coded segments, 13 themes). Overall exit sentiment is mildly negative "
        f"(mean {mean_score:+.2f}) with {neg_pct}% of segments negative. Churn concentrates in support, "
        f"collaboration, roadmap and integrations; onboarding and the core editor are praised. Founders "
        f"exit more negative ({founder:+.2f}) than product managers ({pm:+.2f}). The theme network splits "
        f"into three clusters: API/integrations, cost-and-trust, and scale/collaboration/support. "
        f"Recommendations: a support SLA with a public bug roadmap, document-level permissions, "
        f"stable APIs and transparent pricing."
    )

    story += [
        P("NovaNote - Voice of the Customer", h1),
        P("Exit interview analysis | portfolio project A1", sub),
        HRFlowable(width="100%", thickness=1, color=BLACK, spaceAfter=10),

        P("1. Executive summary", h2),
        P(execsum, body),
        table(pd.DataFrame({"metric": ["segments", "interviews", "themes", "mean sentiment",
                                       "segments negative", "founders mean", "PMs mean"],
                            "value": [n_seg, 10, 13, mean_score, f"{neg_pct}%", founder, pm]}),
             widths=[7 * cm, 5 * cm]),
        Spacer(1, 6),

        P("2. Business problem", h2),
        P("NovaNote (a Notion competitor) is seeing above-expected churn among small and mid-size "
          "business accounts. Leadership needs evidence on which product and service dimensions drive "
          "cancellations, which segments are most negative, and what a retention roadmap should "
          "prioritize first.", body),

        P("3. Data analysis strategy", h2),
        P("An interpretive-to-quantitative pipeline was applied: (1) convert interviews into a coded "
          "segmented corpus; (2) measure sentiment robustly with a pretrained transformer cross-checked "
          "against a lexicon model and human gold labels; (3) model theme relationships as a weighted "
          "co-occurrence network with centrality, community detection and propagation indicators; "
          "(4) surface everything in an interactive dashboard and a written report.", body),

        P("4. Technological stack (all free / open-source)", h2),
        table(pd.DataFrame({
            "tool": ["Python 3.12", "pandas / numpy", "transformers + RoBERTa", "VADER (NLTK)",
                     "networkx + python-louvain", "scipy / scikit-learn", "matplotlib / wordcloud",
                     "Streamlit + Plotly", "reportlab", "GitHub"],
            "role": ["processing", "data handling", "sentiment model", "cross-check",
                     "graph, centrality, communities", "metrics / stats", "static figures",
                     "interactive dashboard", "PDF report", "hosting & versioning"]}),
             widths=[7 * cm, 9.5 * cm]),
        Spacer(1, 6),

        P("5. Methodology", h2),
        P("Corpus: 10 fictional-but-realistic exit interviews (5 founders, 5 product managers), split "
          "into 50 paragraph segments. Each segment was coded multi-label against a 13-theme dictionary "
          "using regex keyword matching.", body),
        P("Sentiment: per segment, cardiffnlp/twitter-roberta-base-sentiment (CPU) returns a "
          "positive/neutral/negative distribution scored [-1, +1]; VADER supplies a lexicon baseline. "
          "Reliability is reported against a hand-labeled gold set (accuracy, macro-F1, Cohen's kappa) "
          "and model agreement is audited via Cohen's kappa.", body),
        P("Network: a weighted co-occurrence graph connects themes that co-occur in the same segment. "
          "Centrality (degree, weighted degree, betweenness, closeness, eigenvector), PageRank as an "
          "influence proxy, also strong-tie reach. Communities were found with Louvain. Layout is "
          "a reproducible 3D force-directed embedding.", body),

        P("6. Results", h2),

        P("6.1 Sentiment model reliability", h3lg()),
        table(rel[["model", "accuracy", "macro_f1", "cohen_kappa"]], widths=[4 * cm, 4 * cm, 4 * cm, 4 * cm]),
        Spacer(1, 4),
        P("RoBERTa outperforms the lexicon baseline on every metric and is retained as the primary "
          "engine. The confusion matrix against gold labels:", body),
        table(cm_, widths=[4.5 * cm] + [3 * cm] * 3),
        Spacer(1, 6),

        P("6.2 Sentiment by theme", h3lg()),
        img("sentiment_by_theme.png"),
        table(senti.sort_values("mean_score")
              [["theme", "mean_score", "n_segments", "negative_share"]].round(2),
              widths=[4.5 * cm, 4 * cm, 4 * cm, 4 * cm]),
        Spacer(1, 4),
        P(f"Most negative: {', '.join(least['theme'])}. Most positive: {', '.join(best['theme'])}.", body),
        Spacer(1, 6),

        P("6.3 Sentiment trends", h3lg()),
        img("sentiment_timeline.png"),
        img("sentiment_by_persona.png"),
        table(month, widths=[6 * cm, 6 * cm]),
        Spacer(1, 6),

        P("6.4 Theme network, centrality and communities", h3lg()),
        img("theme_network_2d.png"),
        table(metrics.sort_values("betweenness", ascending=False)
              [["theme", "degree", "weighted_degree", "betweenness", "closeness", "eigenvector", "pagerank", "community"]],
              widths=[2.6 * cm, 1.9 * cm, 2.6 * cm, 2.4 * cm, 2.2 * cm, 2.3 * cm, 2.0 * cm, 1.9 * cm]),
        Spacer(1, 4),
        P("Communities (Louvain):", body),
        table(communities, widths=[3 * cm, 13.5 * cm]),
        Spacer(1, 4),
        img("theme_correlation_heatmap.png"),
        Spacer(1, 6),

        P("7. Recommendations", h2),
        *[P(f"CLUSTER {comm} [{communities[communities['community'] == comm]['themes'].iloc[0]}]: "
            f"{ACTIONS.get(int(comm), 'deep-dive this cluster.')}", body)
          for comm in communities["community"]],
        Spacer(1, 4),
        P("Priority sequence: (1) support SLA + public roadmap, (2) pricing transparency, "
          "(3) performance at scale, (4) collaboration permissions, (5) API stabilization.",
          body),
    ]
    doc = SimpleDocTemplate(str(DOCS / "A1_report.pdf"), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=1.9 * cm, title="NovaNote Voice of the Customer")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("report written:", DOCS / "A1_report.pdf")


def h3lg():
    return ParagraphStyle("h3lg", fontName="Mono-B", fontSize=9.6, leading=13, textColor=BLACK,
                          spaceBefore=8, spaceAfter=4)


if __name__ == "__main__":
    build()