from typing import Any, Protocol
from itertools import combinations

import numpy as np
import networkx as nx

from typing import Any
import heapq

from src.plotting.graphs import plot_graph, plot_network_via_plotly
from src.common import AnyNxGraph 
import os


class CentralityMeasure(Protocol):
    def __call__(self, G: AnyNxGraph) -> dict[Any, float]:
        ...


def closeness_centrality(G: AnyNxGraph) -> dict[Any, float]:
    res = {}
    for u in G.nodes():
        distances = nx.single_source_dijkstra_path_length(G, u)
        reachable = len(distances) - 1
        total = sum(distances.values())
        if reachable == 0:
            res[u] = 0
        else:
            res[u] = reachable / total 
    return res  


def betweenness_centrality(G: AnyNxGraph) -> dict[Any, float]: 
    n = G.number_of_nodes()
    betweenness = {v: 0.0 for v in G}

    for s in G:
        S = []
        P = {v: [] for v in G}
        sigma = {v: 0.0 for v in G}
        dist = {v: float('inf') for v in G}

        sigma[s] = 1.0
        dist[s] = 0.0
        pq = [(0.0, s)]

        while pq:
            (d, v) = heapq.heappop(pq)
            if d > dist[v]:
                continue
            S.append(v)
            for w, edata in G[v].items():
                weight = edata.get('weight', 1) if isinstance(edata, dict) else 1
                vw_dist = dist[v] + weight
                if dist[w] > vw_dist:
                    dist[w] = vw_dist
                    heapq.heappush(pq, (dist[w], w))
                    sigma[w] = sigma[v]
                    P[w] = [v]
                elif abs(dist[w] - vw_dist) < 1e-12:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = {v: 0.0 for v in G}
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] != 0:
                    coef = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                else:
                    coef = 0.0
                delta[v] += coef
            if w != s:
                betweenness[w] += delta[w]

    if not G.is_directed():
        for v in betweenness:
            betweenness[v] /= 2.0

    if G.is_directed():
        denom = (n - 1) * (n - 2)
    else:
        denom = (n - 1) * (n - 2) / 2.0
    if denom > 0:
        for v in betweenness:
            betweenness[v] /= denom

    return betweenness


def eigenvector_centrality(G, weight='weight', max_iter=100, tol=1e-6):
    nodes = list(G.nodes())
    n = len(nodes)
    A = np.zeros((n, n))
    index = {node: i for i, node in enumerate(nodes)}
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1) if weight else 1
        i, j = index[u], index[v]
        A[i][j] = w
        A[j][i] = w

    x = np.ones(n)
    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    else:
        print("Warning: power iteration did not converge")

    centrality = {node: x[i] for node, i in index.items()}
    return centrality


def plot_centrality_measure(G: AnyNxGraph, measure: CentralityMeasure) -> None:
    values = measure(G)
    if values is not None:
        plot_graph(G, node_weights=values, figsize=(14, 8), name=measure.__name__)
    else:
        print(f"Implement {measure.__name__}")


if __name__ == "__main__":
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "USairport500.txt"),
        os.path.join(os.path.dirname(__file__), "USairport500.txt"),
        os.path.join(os.getcwd(), "practicum_3", "USairport500.txt"),
        os.path.join(os.getcwd(), "USairport500.txt"),
    ]

    data_path = None
    for p in possible_paths:
        p_norm = os.path.normpath(p)
        if os.path.exists(p_norm):
            data_path = p_norm
            break

    if data_path is None:
        print("USairport500.txt not found — строим тестовый граф (karate)")
        G = nx.karate_club_graph()
    else:
        print(f"Loading USairport data from {data_path}")
        G = nx.Graph()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    continue
                w = None
                if len(parts) >= 3:
                    try:
                        w = float(parts[2])
                    except ValueError:
                        w = None
                if w is not None:
                    G.add_edge(u, v, weight=w)
                else:
                    G.add_edge(u, v)

    pos = nx.spring_layout(G)

    try:
        plot_network_via_plotly(G, pos, name="USairport")
        print("Wrote USairport.html via Plotly")
    except Exception as e:
        print("Plotly export failed:", e)

    plot_centrality_measure(G, closeness_centrality)
    plot_centrality_measure(G, betweenness_centrality)
    plot_centrality_measure(G, eigenvector_centrality)

