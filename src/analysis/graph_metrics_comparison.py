

from pathlib import Path
import json
import math
import pandas as pd
import networkx as nx

META_PATH = Path("data/metadata.csv")

ORIG_ROOT = Path("outputs/human_graphs")
SAME_ROOT = Path("outputs/llm_graphs/same_culture")
CROSS_ROOT = Path("outputs/llm_graphs/cross_culture")

OUT_PATH = Path("outputs/graph_metrics_comparison.csv")


def load_graph(graph_path: Path) -> nx.DiGraph:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    G = nx.DiGraph()

    for node in data["nodes"]:
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **attrs)

    for edge in data["edges"]:
        source = edge["source"]
        target = edge["target"]
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        G.add_edge(source, target, **attrs)

    return G


def average_degree(G: nx.DiGraph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    return sum(dict(G.degree()).values()) / G.number_of_nodes()


def longest_path_length_safe(G: nx.DiGraph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    try:
        if nx.is_directed_acyclic_graph(G):
            return nx.dag_longest_path_length(G)
        return 0
    except Exception:
        return 0


def basic_graph_metrics(G: nx.DiGraph) -> dict:
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
        "avg_degree": average_degree(G),
        "longest_path": longest_path_length_safe(G),
    }


def safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return math.nan
    return a / b


def optional_graph_edit_distance(G1: nx.DiGraph, G2: nx.DiGraph, max_nodes: int = 80):
    """
    GED is slow. Only compute for small graphs.
    """
    if G1.number_of_nodes() > max_nodes or G2.number_of_nodes() > max_nodes:
        return math.nan
    try:
        return nx.graph_edit_distance(G1, G2)
    except Exception:
        return math.nan


def compare_pair(original_path: Path, retelling_path: Path, condition: str, story_id: str, culture: str) -> dict:
    G_orig = load_graph(original_path)
    G_ret = load_graph(retelling_path)

    m_orig = basic_graph_metrics(G_orig)
    m_ret = basic_graph_metrics(G_ret)

    row = {
        "story_id": story_id,
        "culture": culture,
        "condition": condition,

        "orig_nodes": m_orig["num_nodes"],
        "orig_edges": m_orig["num_edges"],
        "orig_density": m_orig["density"],
        "orig_avg_degree": m_orig["avg_degree"],
        "orig_longest_path": m_orig["longest_path"],

        "ret_nodes": m_ret["num_nodes"],
        "ret_edges": m_ret["num_edges"],
        "ret_density": m_ret["density"],
        "ret_avg_degree": m_ret["avg_degree"],
        "ret_longest_path": m_ret["longest_path"],

        "compression_ratio_nodes": safe_ratio(m_ret["num_nodes"], m_orig["num_nodes"]),
        "compression_ratio_edges": safe_ratio(m_ret["num_edges"], m_orig["num_edges"]),
        "longest_path_ratio": safe_ratio(m_ret["longest_path"], m_orig["longest_path"]),

        "density_diff": m_ret["density"] - m_orig["density"],
        "avg_degree_diff": m_ret["avg_degree"] - m_orig["avg_degree"],
        "longest_path_diff": m_ret["longest_path"] - m_orig["longest_path"],
    }

    # optional, slow
    row["graph_edit_distance"] = float("nan")

    return row


def main():
    meta = pd.read_csv(META_PATH)

    rows = []

    for _, r in meta.iterrows():
        story_id = str(r["story_id"])
        culture = str(r["culture"])

        orig_graph = ORIG_ROOT / story_id / "graph.json"
        same_graph = SAME_ROOT / story_id / "graph.json"
        cross_graph = CROSS_ROOT / story_id / "graph.json"

        if orig_graph.exists() and same_graph.exists():
            rows.append(compare_pair(
                original_path=orig_graph,
                retelling_path=same_graph,
                condition="same_culture",
                story_id=story_id,
                culture=culture
            ))

        if orig_graph.exists() and cross_graph.exists():
            rows.append(compare_pair(
                original_path=orig_graph,
                retelling_path=cross_graph,
                condition="cross_culture",
                story_id=story_id,
                culture=culture
            ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()