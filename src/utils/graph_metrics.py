import numpy as np

def compute_cut_size(graph, partitions):
    """Count the number of edges that cross between partitions."""
    if not partitions:
        return 0
    # Convert dict to list if needed
    if isinstance(partitions, dict):
        partitions = list(partitions.values())
    cut = 0
    for p in partitions:
        nodes = p.nodes if hasattr(p, 'nodes') else p
        for node in nodes:
            for neighbor in graph.get_neighbors(node):
                # Check if neighbor is in a different partition
                for q in partitions:
                    if q is p:
                        continue
                    other_nodes = q.nodes if hasattr(q, 'nodes') else q
                    if neighbor in other_nodes:
                        cut += 1
                        break
    return cut // 2  # Each cut counted twice

def compute_balance(partitions):
    """Return min(partition_size)/max(partition_size) as balance metric."""
    if not partitions:
        return 0.0
    # Convert dict to list if needed
    if isinstance(partitions, dict):
        partitions = list(partitions.values())
    # Get partition sizes
    sizes = [len(p.nodes if hasattr(p, 'nodes') else p) for p in partitions]
    if not sizes or max(sizes) == 0:
        return 0.0
    return min(sizes) / max(sizes)
    return min(sizes) / max(sizes)

def compute_conductance(graph, partitions):
    """Return mean conductance across partitions.
    Conductance = cut_edges / min(internal_edges, external_edges)
    Lower conductance is better (fewer edges leaving the partition).
    """
    if not partitions:
        return 0.0
    # Convert dict to list if needed
    if isinstance(partitions, dict):
        partitions = list(partitions.values())
    conductances = []
    for p in partitions:
        nodes = p.nodes if hasattr(p, 'nodes') else p
        if not nodes:  # Skip empty partitions
            continue
        internal_edges = 0
        external_edges = 0
        for node in nodes:
            for neighbor in graph.get_neighbors(node):
                if neighbor in nodes:
                    internal_edges += 1
                else:
                    external_edges += 1
        internal_edges //= 2  # Each internal edge counted twice
        
        # Calculate conductance:
        # - If no edges at all: conductance = 0 (isolated nodes)
        # - If only internal edges: conductance = 0 (perfect partition)
        # - If only external edges: conductance = 1 (worst case)
        # - Otherwise: external_edges / min(internal_edges, external_edges)
        if internal_edges == 0 and external_edges == 0:
            conductances.append(0.0)  # Isolated nodes
        elif internal_edges == 0:
            conductances.append(1.0)  # All edges are external
        elif external_edges == 0:
            conductances.append(0.0)  # All edges are internal
        else:
            conductance = external_edges / (internal_edges + external_edges)
            conductances.append(conductance)
    
    return np.mean(conductances) if conductances else 1.0  # Return worst case if no valid partitions
