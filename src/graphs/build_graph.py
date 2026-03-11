from __future__ import annotations
from typing import List, Dict, Any
import networkx as nx

from .extract_events import Event

def build_event_graph(events: List[Event]) -> nx.DiGraph:
    """
    Nodes: event labels
    Edges: temporal order (next event)
    """
    G = nx.DiGraph()

    # Add nodes with metadata
    for i, e in enumerate(events):
        node_id = f"e{i:04d}"
        G.add_node(
            node_id,
            label=e.label,
            sent_id=e.sent_id,
            sentence=e.sentence,
            subj=e.subj,
            verb=e.verb,
            obj=e.obj,
        )

    # Temporal edges
    for i in range(len(events) - 1):
        G.add_edge(f"e{i:04d}", f"e{i+1:04d}", relation="next")

    return G

def graph_to_jsonable(G: nx.DiGraph) -> Dict[str, Any]:
    nodes = [{"id": n, **G.nodes[n]} for n in G.nodes]
    edges = [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges]
    return {"directed": True, "nodes": nodes, "edges": edges}
