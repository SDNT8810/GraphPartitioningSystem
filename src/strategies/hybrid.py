"""
HybridPartitioningStrategy: Combines spectral and RL-based partitioning.
Applies robust partition management (balancing, merging, splitting) after assignment.

See also:
- [CODEMAP.md](../../CODEMAP.md)
- [TODO.md](../../TODO.md)
- [INDEX.md](../INDEX.md)
"""

from typing import Dict, Set
from .spectral import SpectralPartitioningStrategy
from .dynamic_partitioning import DynamicPartitioning
from ..core.graph import Graph, Partition

class HybridPartitioningStrategy:
    """
    Hybrid partitioning: spectral initialization, RL refinement, robust partition management.
    """
    def __init__(self, config):
        self.config = config
        self.spectral = SpectralPartitioningStrategy(config)
        self.rl = DynamicPartitioning(config)

    def partition(self, graph: Graph) -> Dict[int, Partition]:
        """
        1. Use spectral partitioning to initialize.
        2. Optionally refine with RL-based partitioning.
        3. Apply balancing, merging, and splitting as in other strategies.
        """
        # Pass config to graph
        graph.config = self.config
        # Step 1: Spectral initialization
        partitions = self.spectral.partition(graph)
        # Step 2: RL refinement (optional, can be expanded)
        from src.config.system_config import AgentConfig
        agent_config = getattr(self.config, 'agent_config', None)
        if agent_config is None:
            agent_config = AgentConfig()
        # Convert spectral partitions to graph.partitions format
        graph.partitions = {}
        for pid, partition in partitions.items():
            graph.partitions[pid] = partition
        # Initialize RL with current partitions
        self.rl.initialize(graph, agent_config)
        rl_partitions = self.rl.partition(graph)  # Use partition instead of train
        # Update graph.partitions with RL results
        graph.partitions = {}
        for pid, partition in rl_partitions.items():
            graph.partitions[pid] = partition
        # Step 3: Robust partition management
        # Convert graph.partitions to list format for easier manipulation
        partitions_list = list(graph.partitions.values())
        # Balance if needed
        if len(partitions_list) > 1:
            # Find smallest and largest partitions
            min_size = min(len(p.nodes) if isinstance(p, Partition) else len(p) for p in partitions_list)
            max_size = max(len(p.nodes) if isinstance(p, Partition) else len(p) for p in partitions_list)
            target_size = graph.num_nodes // len(partitions_list)
            # Merge small partitions
            if min_size < target_size // 2:
                small_partition = [p for p in partitions_list if (len(p.nodes) if isinstance(p, Partition) else len(p)) == min_size][0]
                other_partition = [p for p in partitions_list if p.id != small_partition.id][0]
                # Move nodes from small to other partition
                nodes = small_partition.nodes if isinstance(small_partition, Partition) else small_partition
                for node in list(nodes):
                    if isinstance(small_partition, Partition):
                        small_partition.remove_node(node)
                    else:
                        small_partition.remove(node)
                    if isinstance(other_partition, Partition):
                        other_partition.add_node(node)
                    else:
                        other_partition.add(node)
                # Remove small partition
                partitions_list.remove(small_partition)
                # Update graph.partitions
                graph.partitions = {p.id: p for p in partitions_list}
            # Split large partitions
            elif max_size > 2 * target_size:
                large_partition = [p for p in partitions_list if (len(p.nodes) if isinstance(p, Partition) else len(p)) == max_size][0]
                # Create new partition
                new_partition = Partition(id=max(p.id for p in partitions_list) + 1)
                # Move half the nodes to new partition
                nodes = large_partition.nodes if isinstance(large_partition, Partition) else large_partition
                nodes_to_move = list(nodes)[:len(nodes)//2]
                for node in nodes_to_move:
                    if isinstance(large_partition, Partition):
                        large_partition.remove_node(node)
                    else:
                        large_partition.remove(node)
                    new_partition.add_node(node)
                # Add new partition
                partitions_list.append(new_partition)
                # Update graph.partitions
                graph.partitions = {p.id: p for p in partitions_list}
        # Return current partitions
        if hasattr(graph, 'partitions'):
            # Convert any sets to Partition objects
            result = {}
            for pid, p in graph.partitions.items():
                if isinstance(p, Partition):
                    result[pid] = p
                else:
                    # Convert set to Partition
                    partition = Partition(id=pid)
                    partition.nodes = p
                    result[pid] = partition
            return result
        return partitions