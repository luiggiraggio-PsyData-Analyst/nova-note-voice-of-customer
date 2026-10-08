import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvain

from project_config import PROCESSED, FIGURES, TABLES, NETWORK, THEMES

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


def main():
    segs = pd.read_csv(TABLES / "segment_sentiment.csv")
    matrix = segs[[f"theme_{t}" for t in THEMES]].values.astype(int)
    cooc = matrix.T @ matrix
    np.fill_diagonal(cooc, 0)

    G = nx.Graph()
    G.add_nodes_from(THEMES)
    for i, a in enumerate(THEMES):
        for j, b in enumerate(THEMES):
            if i < j and cooc[i, j] > 0:
                G.add_edge(a, b, weight=int(cooc[i, j]))

    edges = []
    for a, b, d in G.edges(data=True):
        ia, ib = THEMES.index(a), THEMES.index(b)
        inter = int(cooc[ia, ib])
        union = int((matrix[:, ia] | matrix[:, ib]).sum())
        edges.append({"source": a, "target": b, "weight": inter,
                      "jaccard": round(inter / union, 3) if union else 0})
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(TABLES / "theme_edges.csv", index=False)

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)

    deg = dict(G.degree())
    w_deg = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)
    eigen = nx.eigenvector_centrality_numpy(G, weight="weight")
    pagerank = nx.pagerank(G, weight="weight")

    max_w = max(d.get("weight", 0) for _, _, d in G.edges(data=True)) or 1
    strong = nx.Graph()
    strong.add_nodes_from(THEMES)
    strong.add_edges_from((a, b) for a, b, d in G.edges(data=True)
                          if d.get("weight", 0) >= 0.75 * max_w)
    reach = {n: len(nx.node_connected_component(strong, n)) for n in THEMES}

    nodes = pd.DataFrame({
        "theme": THEMES,
        "degree": [deg[t] for t in THEMES],
        "weighted_degree": [round(w_deg[t], 2) for t in THEMES],
        "betweenness": [round(betweenness[t], 3) for t in THEMES],
        "closeness": [round(closeness[t], 3) for t in THEMES],
        "eigenvector": [round(eigen[t], 3) for t in THEMES],
        "pagerank": [round(pagerank[t], 3) for t in THEMES],
        "community": [partition[t] for t in THEMES],
        "strong_reach": [reach[t] for t in THEMES],
    })
    nodes.to_csv(TABLES / "theme_metrics.csv", index=False)
    communities = nodes.groupby("community")["theme"].apply(lambda s: ", ".join(s)).reset_index()
    communities.columns = ["community", "themes"]
    communities.to_csv(TABLES / "communities.csv", index=False)

    corr = pd.DataFrame(matrix, columns=THEMES).corr()
    corr.to_csv(TABLES / "theme_correlation.csv")

    layout = nx.spring_layout(G, dim=3, seed=42, weight="weight", k=0.9, iterations=200)
    with open(NETWORK / "layout3d.json", "w") as fh:
        json.dump({t: layout[t].tolist() for t in THEMES}, fh)
    nx.write_gml(G, NETWORK / "theme_graph.gml")

    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=2, weight="weight", k=0.7)
    cmap_vals = np.asarray([partition[t] for t in THEMES])
    colors = plt.cm.Greys(np.linspace(0.25, 0.9, len(set(partition.values()))))
    node_colors = [colors[c] for c in cmap_vals]
    sizes = [300 + 80 * betweenness[t] * 10 for t in THEMES]
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.6 * d.get("weight", 1) for _, _, d in G.edges(data=True)],
                           edge_color="#b0b0b0")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors,
                           edgecolors="#111111", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_family="DejaVu Sans Mono", font_size=9)
    ax.set_title("theme co-occurrence network (communities shaded)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "theme_network_2d.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.4))
    im = ax.imshow(corr.values, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(range(len(THEMES)))
    ax.set_xticklabels(THEMES, rotation=90)
    ax.set_yticks(range(len(THEMES)))
    ax.set_yticklabels(THEMES)
    for i in range(len(THEMES)):
        for j in range(len(THEMES)):
            if i != j:
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        color="#111111", fontsize=6)
    ax.set_title("theme co-occurrence correlation")
    fig.colorbar(im, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "theme_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    top = nodes.nlargest(8, "betweenness")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top["theme"], top["betweenness"], color="#2f2f2f", edgecolor="#111111", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("betweenness centrality")
    ax.set_title("top themes by betweenness")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "centrality_top.png", dpi=180)
    plt.close(fig)

    print(nodes.sort_values("betweenness", ascending=False).to_string(index=False))
    print("\ncommunities:")
    print(communities.to_string(index=False))


if __name__ == "__main__":
    main()