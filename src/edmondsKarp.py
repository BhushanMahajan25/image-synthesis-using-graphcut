# Edmonds-Karp algorithm for maximum flow problems.

from residualFlowNetwork import build_residual_network

__all__ = ["edmonds_karp"]


def edmonds_karp_core(residual_graph, source, sink, cutoff):
    residual_graph_nodes = residual_graph.nodes
    residual_graph_pred = residual_graph.pred
    residual_graph_succ = residual_graph.succ

    inf = residual_graph.graph["inf"]

    def augment(path):
        # Augment flow along a path from source to sink.
        # Determine the path residual capacity.
        flow = inf
        it = iter(path)
        u = next(it)
        for v in it:
            attr = residual_graph_succ[u][v]
            flow = min(flow, attr["capacity"] - attr["flow"])
            u = v
        # Augment flow along the path.
        it = iter(path)
        u = next(it)
        for v in it:
            residual_graph_succ[u][v]["flow"] += flow
            residual_graph_succ[v][u]["flow"] -= flow
            u = v
        return flow

    def bidirectional_bfs():
        # Bidirectional breadth-first search for an augmenting path.
        pred = {source: None}
        q_s = [source]
        succ = {sink: None}
        q_t = [sink]
        while True:
            q = []
            if len(q_s) <= len(q_t):
                for u in q_s:
                    for v, attr in residual_graph_succ[u].items():
                        if v not in pred and attr["flow"] < attr["capacity"]:
                            pred[v] = u
                            if v in succ:
                                return v, pred, succ
                            q.append(v)
                if not q:
                    return None, None, None
                q_s = q
            else:
                for u in q_t:
                    for v, attr in residual_graph_pred[u].items():
                        if v not in succ and attr["flow"] < attr["capacity"]:
                            succ[v] = u
                            if v in pred:
                                return v, pred, succ
                            q.append(v)
                if not q:
                    return None, None, None
                q_t = q

    # Look for shortest augmenting paths using breadth-first search.
    flow_value = 0
    while flow_value < cutoff:
        v, pred, succ = bidirectional_bfs()
        if pred is None:
            break
        path = [v]
        # Trace a path from source to v.
        u = v
        while u != source:
            u = pred[u]
            path.append(u)
        path.reverse()
        # Trace a path from v to sink.
        u = v
        while u != sink:
            u = succ[u]
            path.append(u)
        flow_value += augment(path)

    return flow_value


def edmonds_karp_impl(G, source, sink, capacity, residual, cutoff):
    if residual is None:
        residual_graph = build_residual_network(G, capacity)
    else:
        residual_graph = residual

    # Initialize/reset the residual network.
    for u in residual_graph:
        for e in residual_graph[u].values():
            e["flow"] = 0

    if cutoff is None:
        cutoff = float("inf")
    residual_graph.graph["flow_value"] = edmonds_karp_core(residual_graph, source, sink, cutoff)

    return residual_graph


def edmonds_karp(input_graph,source,sink,capacity="capacity",residual=None,value_only=False,cutoff=None):
    residual_graph = edmonds_karp_impl(input_graph, source, sink, capacity, residual, cutoff)
    residual_graph.graph["algorithm"] = "edmonds_karp"
    return residual_graph