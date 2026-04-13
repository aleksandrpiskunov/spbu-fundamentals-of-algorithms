from pathlib import Path
import heapq
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx

from practicum_4.dfs import GraphTraversal 
from src.plotting.graphs import plot_graph
from src.common import AnyNxGraph


class DijkstraAlgorithm(GraphTraversal):
    def __init__(self, G: AnyNxGraph) -> None:
        self.shortest_paths: dict[Any, list[Any]] = {}
        super().__init__(G)

    def previsit(self, node: Any, **params) -> None:
        """List of params:
        * path: list[Any] (path from the initial node to the given node)
        """
        self.shortest_paths[node] = params["path"]

    def postvisit(self, node: Any, **params) -> None:
        pass

    def run(self, node: Any) -> None:
        self.reset()
        self.shortest_paths.clear()

        priority_queue: list[tuple[float, Any, list[Any]]] = []
        heapq.heappush(priority_queue, (0.0, node, [node]))

        while len(priority_queue) > 0:
            current_distance, current_node, current_path = heapq.heappop(priority_queue)

            if current_node in self.visited:
                continue

            self.visited.add(current_node)
            self.previsit(current_node, path=current_path)

            for neigh in self.G.neighbors(current_node):
                if neigh in self.visited:
                    continue

                edge_weight = self.G[current_node][neigh]["weight"]
                new_distance = current_distance + edge_weight
                new_path = current_path + [neigh]
                heapq.heappush(priority_queue, (new_distance, neigh, new_path))



if __name__ == "__main__":
    G = nx.read_edgelist(
        Path("practicum_4") / "simple_weighted_graph_9_nodes.edgelist",
        create_using=nx.Graph
    )
    plot_graph(G)

    sp = DijkstraAlgorithm(G)
    sp.run("0")

    test_node = "5"
    shortest_path_edges = [
        (sp.shortest_paths[test_node][i], sp.shortest_paths[test_node][i + 1])
        for i in range(len(sp.shortest_paths[test_node]) - 1)
    ]
    plot_graph(G, highlighted_edges=shortest_path_edges)

