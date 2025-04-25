import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from .base_agent import BaseAgent
from .local_agent import LocalAgent
from ..core.graph import Graph, Partition
from ..config.system_config import AgentConfig, PartitionConfig

class GlobalAgent(BaseAgent):
    """Global agent for coordinating local agents and implementing hybrid partitioning."""
    
    def __init__(self, config: AgentConfig, partition_config: PartitionConfig):
        super().__init__(node_id=-1, config=config)  # Global agent has no node ID
        self.partition_config = partition_config
        self.local_agents: Dict[int, LocalAgent] = {}
        self.partition_history: List[Dict[int, Set[int]]] = []
        self.metrics_history: List[Dict] = []
        
    def register_local_agent(self, agent: LocalAgent) -> None:
        """Register a local agent with the global agent."""
        self.local_agents[agent.node_id] = agent
        
    def coordinate_partitioning(self, graph: Graph) -> Dict[int, int]:
        """Coordinate the partitioning process among local agents."""
        # Collect preferences from all agents
        preferences = {}
        for agent_id, agent in self.local_agents.items():
            preferences[agent_id] = agent.get_partition_preference(graph)
            
        # Initialize partition assignments
        assignments = {}
        partition_sizes = defaultdict(int)
        
        # First pass: assign nodes based on preferences
        for agent_id, pref in preferences.items():
            if not pref:
                continue
                
            # Get top preferred partitions
            sorted_pref = sorted(pref.items(), key=lambda x: x[1], reverse=True)
            for partition_id, score in sorted_pref:
                if partition_sizes[partition_id] < self.partition_config.max_partition_size:
                    assignments[agent_id] = partition_id
                    partition_sizes[partition_id] += 1
                    break
                    
        # Second pass: handle unassigned nodes
        unassigned = set(self.local_agents.keys()) - set(assignments.keys())
        for agent_id in unassigned:
            # Find partition with minimum size
            min_partition = min(partition_sizes.items(), key=lambda x: x[1])[0]
            assignments[agent_id] = min_partition
            partition_sizes[min_partition] += 1
            
        return assignments
        
    def update_partitions(self, graph: Graph, assignments: Dict[int, int]) -> None:
        """Update partition assignments in the graph."""
        # Store current partition state
        current_partitions = {
            pid: set(part.nodes) for pid, part in graph.partitions.items()
        }
        self.partition_history.append(current_partitions)
        
        # Update partitions
        for agent_id, partition_id in assignments.items():
            current_partition = self.local_agents[agent_id].current_partition
            if current_partition != partition_id:
                graph.move_node(agent_id, current_partition, partition_id)
                self.local_agents[agent_id].update_partition(partition_id)
                
        # Store metrics
        metrics = {
            'partition_sizes': {pid: len(part.nodes) for pid, part in graph.partitions.items()},
            'densities': {pid: part.density for pid, part in graph.partitions.items()},
            'conductances': {pid: part.conductance for pid, part in graph.partitions.items()}
        }
        self.metrics_history.append(metrics)
        
    def evaluate_partitioning(self, graph: Graph) -> Dict[str, float]:
        """Evaluate the quality of the current partitioning."""
        metrics = graph.get_partition_metrics()
        
        # Calculate balance score
        sizes = [len(part.nodes) for part in graph.partitions.values()]
        balance_score = 1 - (max(sizes) - min(sizes)) / graph.num_nodes
        
        # Calculate average density and conductance
        avg_density = np.mean([m['density'] for m in metrics.values()])
        avg_conductance = np.mean([m['conductance'] for m in metrics.values()])
        
        # Calculate cut size
        cut_size = 0
        for u in range(graph.num_nodes):
            for v in graph.get_neighbors(u):
                if u < v and self.local_agents[u].current_partition != self.local_agents[v].current_partition:
                    cut_size += 1
                    
        return {
            'balance_score': balance_score,
            'avg_density': avg_density,
            'avg_conductance': avg_conductance,
            'cut_size': cut_size
        }
        
    def get_recovery_points(self) -> List[Dict[int, Set[int]]]:
        """Get historical partition states for recovery."""
        return self.partition_history
        
    def get_metrics_history(self) -> List[Dict]:
        """Get historical metrics for analysis."""
        return self.metrics_history
        
    def save_state(self, path: str) -> None:
        """Save the global agent's state."""
        state = {
            'partition_history': self.partition_history,
            'metrics_history': self.metrics_history
        }
        torch.save(state, path)
        
    def load_state(self, path: str) -> None:
        """Load the global agent's state."""
        state = torch.load(path)
        self.partition_history = state['partition_history']
        self.metrics_history = state['metrics_history'] 