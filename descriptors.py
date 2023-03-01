import numpy as np
from networkit.centrality import Betweenness
from networkit.distance import APSP
from networkit.graph import Graph
from networkit.linkprediction import JaccardIndex
from networkit.sparsification import LocalDegreeScore


def calculate_shortest_paths(graph: Graph) -> np.array:
    # Networkit is faster than NetworkX for large graphs
    apsp = APSP(graph)
    apsp.run()
    path_lengths = apsp.getDistances(asarray=True)

    path_lengths = path_lengths.ravel()

    # filter out 0 length "paths" from node to itself
    path_lengths = path_lengths[np.nonzero(path_lengths)]

    # Networkit assigns extremely high values (~1e308) to mark infinite
    # distances for disconnected components, so we simply drop them
    path_lengths = path_lengths[path_lengths < 1e100]

    return path_lengths


def calculate_edge_betweenness(graph: Graph) -> np.ndarray:
    betweeness = Betweenness(graph, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    scores = np.array(scores, dtype=np.float32)
    return scores


def calculate_jaccard_index(graph: Graph) -> np.ndarray:
    jaccard_index = JaccardIndex(graph)
    scores = [jaccard_index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    return scores


def calculate_local_degree_score(graph: Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, dtype=np.float32)
