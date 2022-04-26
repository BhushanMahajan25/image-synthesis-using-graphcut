# Residual network flow implementation

import networkx as nx

def build_residual_network(input_graph, capacity):
    #Build a residual network and initialize a zero flow.

    residual_graph = nx.DiGraph()
    residual_graph.add_nodes_from(input_graph)

    inf = float("inf")

    # Extract edges with positive capacities. Self loops excluded.
    edge_list = [(u, v, attr) for u, v, attr in input_graph.edges(data=True) if u != v and attr.get(capacity, inf) > 0]
    # Simulate infinity with three times the sum of the finite edge capacities
    # or any positive value if the sum is zero. This allows the
    # infinite-capacity edges to be distinguished for unboundedness detection
    # and directly participate in residual capacity calculation. If the maximum
    # flow is finite, these edges cannot appear in the minimum cut and thus
    # guarantee correctness. Since the residual capacity of an
    # infinite-capacity edge is always at least 2/3 of inf, while that of an
    # finite-capacity edge is at most 1/3 of inf, if an operation moves more
    # than 1/3 of inf units of flow to t, there must be an infinite-capacity
    # s-t path in G.
    inf = (3 * sum(attr[capacity] for u, v, attr in edge_list if capacity in attr and attr[capacity] != inf) or 1)
    if input_graph.is_directed():
        for u, v, attr in edge_list:
            r = min(attr.get(capacity, inf), inf)
            if not residual_graph.has_edge(u, v):
                # Both (u, v) and (v, u) must be present in the residual network.
                residual_graph.add_edge(u, v, capacity=r)
                residual_graph.add_edge(v, u, capacity=0)
            else:
                # The edge (u, v) was added when (v, u) was visited.
                residual_graph[u][v]["capacity"] = r
    else:
        for u, v, attr in edge_list:
            # Add a pair of edges with equal residual capacities.
            r = min(attr.get(capacity, inf), inf)
            residual_graph.add_edge(u, v, capacity=r)
            residual_graph.add_edge(v, u, capacity=r)

    # Record the value simulating infinity.
    residual_graph.graph["inf"] = inf

    return residual_graph