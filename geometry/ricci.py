"""
Ollivier-Ricci curvature on a kNN graph, implemented directly.

Definition (Ollivier 2009): for nodes x, y connected by an edge,
    kappa(x, y) = 1 - W_1(mu_x, mu_y) / d(x, y)
where mu_x is a probability measure on the neighborhood of x, W_1 is the
Wasserstein-1 distance between those measures, and d(x, y) is the graph
distance (edge length).

Interpretation:
    kappa > 0  : geodesics from x and y converge (positive curvature,
                 basin / attracting region)
    kappa = 0  : locally flat
    kappa < 0  : geodesics diverge (negative curvature, saddle or escape)

Per-node scalar curvature is the mean Ollivier-Ricci over that node's
incident edges. This is the scalar we use as the "accelerometer reading."

We implement directly (not via the GraphRicciCurvature package) because
that package depends on networkit which requires building from source
with cmake — a dependency we don't need for ~50 lines of math.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import ot  # POT: Python Optimal Transport


def local_probability_measure(
    G: nx.Graph,
    node: int,
    alpha: float,
) -> tuple[np.ndarray, list[int]]:
    """Return (probability vector, support node list) for the measure at node.

    Standard Ollivier-Ricci: mass alpha on the node, (1-alpha) uniformly on
    the immediate neighborhood. Neighbors are 1-hop; we do not go further
    because that blurs local geometry.
    """
    neighbors = list(G.neighbors(node))
    n_nbrs = len(neighbors)

    if n_nbrs == 0:
        # Isolated node; return point mass on itself.
        return np.array([1.0]), [node]

    support = [node] + neighbors
    probs = np.empty(len(support))
    probs[0] = alpha
    probs[1:] = (1.0 - alpha) / n_nbrs
    return probs, support


def edge_ollivier_ricci(
    G: nx.Graph,
    u: int,
    v: int,
    alpha: float,
) -> float:
    """Compute Ollivier-Ricci curvature on edge (u, v).

    Requires edges to carry a 'weight' attribute representing metric
    distance between node feature vectors (Euclidean in feature space).
    """
    p_u, support_u = local_probability_measure(G, u, alpha)
    p_v, support_v = local_probability_measure(G, v, alpha)

    # Build the ground-distance cost matrix between the two supports.
    # Use Euclidean distance between node feature vectors (the 'pos' attr).
    pos_u = np.array([G.nodes[n]["pos"] for n in support_u])
    pos_v = np.array([G.nodes[n]["pos"] for n in support_v])

    # Pairwise distances: (len(support_u), len(support_v))
    cost = np.linalg.norm(
        pos_u[:, None, :] - pos_v[None, :, :], axis=2
    )

    # Wasserstein-1 distance via linear programming (exact).
    w1 = ot.emd2(p_u, p_v, cost)

    edge_weight = G[u][v]["weight"]
    if edge_weight <= 0:
        return 0.0

    return 1.0 - w1 / edge_weight


def compute_all_edge_curvatures(
    G: nx.Graph,
    alpha: float = 0.5,
) -> dict[tuple[int, int], float]:
    """Compute Ollivier-Ricci on every edge of G.

    Returns dict mapping (u, v) -> kappa. (u, v) is stored in sorted order
    for undirected graphs so lookup is deterministic.
    """
    curvatures = {}
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        curvatures[key] = edge_ollivier_ricci(G, u, v, alpha)
    return curvatures


def scalar_curvature_per_node(
    G: nx.Graph,
    edge_curvatures: dict[tuple[int, int], float],
) -> np.ndarray:
    """Aggregate edge Ollivier-Ricci to a scalar per node (mean over
    incident edges).

    Returns array indexed by node id (assumes nodes are 0..n-1).
    """
    n = G.number_of_nodes()
    scalars = np.zeros(n)
    counts = np.zeros(n)
    for (u, v), kappa in edge_curvatures.items():
        scalars[u] += kappa
        scalars[v] += kappa
        counts[u] += 1
        counts[v] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        scalars = np.where(counts > 0, scalars / counts, 0.0)
    return scalars


def build_knn_graph(
    points: np.ndarray,
    k: int,
) -> nx.Graph:
    """Build a symmetric kNN graph over points (n_samples, n_features).

    Each node n carries 'pos' = its feature vector.
    Each edge (u, v) carries 'weight' = Euclidean distance in feature space.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(points)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(points)
    distances, indices = nbrs.kneighbors(points)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=points[i])

    for i in range(n):
        # indices[i, 0] is i itself; skip it.
        for rank in range(1, k + 1):
            j = int(indices[i, rank])
            d = float(distances[i, rank])
            if G.has_edge(i, j):
                # already present; keep the shorter weight
                if d < G[i][j]["weight"]:
                    G[i][j]["weight"] = d
            else:
                G.add_edge(i, j, weight=d)
    return G


def local_curvature_at_test_point(
    test_point: np.ndarray,
    healthy_points: np.ndarray,
    k: int,
    alpha: float = 0.5,
) -> float:
    """Probe: drop test_point into the local healthy manifold, compute the
    scalar curvature at that node.

    Constructs a small subgraph of test_point + its k nearest healthy neighbors
    + their k nearest healthy neighbors. Computes Ollivier-Ricci on edges
    incident to test_point. Returns mean of those edge curvatures.

    This is Option B from the design discussion: estimate the local curvature
    of the healthy manifold at the query location, without recomputing the
    entire graph.
    """
    from sklearn.neighbors import NearestNeighbors

    # Find k nearest healthy points to the test point.
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(healthy_points)
    test_dists, test_idx = nbrs.kneighbors(test_point.reshape(1, -1))
    test_dists = test_dists[0]
    test_idx = test_idx[0]

    # Expand: for each neighbor of the test point, also include ITS k nearest
    # healthy neighbors. This gives the test-point's node enough local
    # structure that Ollivier-Ricci is well-defined.
    expanded = set(int(i) for i in test_idx)
    for i in test_idx:
        _, nbrs_of_i = nbrs.kneighbors(healthy_points[i].reshape(1, -1))
        for j in nbrs_of_i[0]:
            expanded.add(int(j))
    expanded_list = sorted(expanded)

    # Rebuild a local graph with the test point as node 0 and the expanded
    # healthy neighborhood as nodes 1..m.
    G = nx.Graph()
    G.add_node(0, pos=test_point)
    for local_i, global_i in enumerate(expanded_list, start=1):
        G.add_node(local_i, pos=healthy_points[global_i])

    # Add test-point edges (to its k nearest healthy neighbors).
    global_to_local = {g: l for l, g in enumerate(expanded_list, start=1)}
    for rank, global_i in enumerate(test_idx):
        local_i = global_to_local[int(global_i)]
        G.add_edge(0, local_i, weight=float(test_dists[rank]))

    # Add healthy-healthy edges by running kNN within the expanded subset.
    sub_points = np.array([healthy_points[i] for i in expanded_list])
    sub_nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, len(sub_points)), algorithm="ball_tree"
    ).fit(sub_points)
    sub_d, sub_idx = sub_nbrs.kneighbors(sub_points)
    for local_i in range(len(sub_points)):
        for rank in range(1, sub_idx.shape[1]):
            local_j = int(sub_idx[local_i, rank])
            d = float(sub_d[local_i, rank])
            # local_i and local_j are indices into sub_points; shift by +1 for
            # the test point being at node 0.
            u, v = local_i + 1, local_j + 1
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=d)

    # Compute Ollivier-Ricci on edges incident to the test point only.
    kappas = []
    for neighbor in G.neighbors(0):
        kappas.append(edge_ollivier_ricci(G, 0, neighbor, alpha))
    if not kappas:
        return 0.0
    return float(np.mean(kappas))
