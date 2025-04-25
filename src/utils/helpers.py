# src/utils/helpers.py
"""
Stub for helpers.py. Replace with actual utility functions as needed.
"""

def generate_random_graph(*args, **kwargs):
    """
    Stub for generate_random_graph. Replace with actual implementation.
    """
    raise NotImplementedError("generate_random_graph() is not yet implemented.")

def compute_graph_metrics(graph, partitions):
    """
    Compute basic metrics for the graph and its partitions.
    Args:
        graph: Graph object
        partitions: List of Partition objects
    Returns:
        dict with metrics: num_nodes, num_edges, density, avg_clustering, diameter, partition_sizes, densities, conductances
    """
    import networkx as nx
    num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else len(getattr(graph, 'nodes', []))
    # Count edges (assuming undirected, adjacency matrix)
    if hasattr(graph, 'adj_matrix'):
        num_edges = int(graph.adj_matrix.nonzero().size(0) // 2)
    else:
        num_edges = 0
    # Convert to networkx for advanced metrics
    if hasattr(graph, 'to_networkx'):
        nx_graph = graph.to_networkx()
        density = nx.density(nx_graph)
        avg_clustering = nx.average_clustering(nx_graph)
        try:
            diameter = nx.diameter(nx_graph)
        except nx.NetworkXError:
            # Graph not connected
            diameter = float('inf')
    else:
        density = None
        avg_clustering = None
        diameter = None
    partition_sizes = [len(p.nodes) for p in partitions]
    densities = [getattr(p, 'density', None) for p in partitions]
    conductances = [getattr(p, 'conductance', None) for p in partitions]
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_clustering': avg_clustering,
        'diameter': diameter,
        'partition_sizes': partition_sizes,
        'densities': densities,
        'conductances': conductances
    }

def example_helper():
    """Example helper function."""
    pass
