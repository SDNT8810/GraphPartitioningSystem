# src/strategies/dynamic_partitioning.py

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.local_agent import *
from src.config.system_config import *
from src.core.graph import *
from pathlib import Path
from src.utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance
from src.utils.visualization import TrainingVisualizer, plot_training_progress

class DynamicPartitioning:
    """
    RL-based dynamic partitioning for self-partitioning graphs (autonomous node-level agents).
    Each node is controlled by a LocalAgent that selects partition assignments.
    """
    def __init__(self, config: AgentConfig, experiment_name: str = "experiment"):
        """Initialize the dynamic partitioning strategy."""
        self.config = config
        self.experiment_name = experiment_name
        self.graph = None
        self.local_agents = []
        self.partitions = []
        self.rewards = []
        self.last_stats = None

    def initialize(self, graph, agent_config):
        # Update graph config
        graph.config = agent_config
        self.graph = graph
        
        # Determine number of partitions
        num_partitions = getattr(self.config, 'num_partitions', 2) if self.config else 2
        
        # Update agent config to match number of partitions
        agent_config.action_dim = num_partitions
        
        # Initialize agents
        self.local_agents = [LocalAgent(agent_config, graph, node_id) for node_id in range(graph.num_nodes)]
        
        # Initialize or use existing partitions
        if not graph.partitions:
            # Create new partitions
            for i in range(num_partitions):
                graph.add_partition(i)
            
            # Initial random assignment
            nodes = list(range(graph.num_nodes))
            np.random.shuffle(nodes)
            for i, node in enumerate(nodes):
                partition_id = i % num_partitions
                graph.move_node(node, None, partition_id)
        
        # Update our local partition list
        self.partitions = list(graph.partitions.values())

    def train(self):
        """
        Minimal RL training loop: each agent selects a partition, environment updates, reward assigned.
        Returns stats for integration/testing.
        """
        if self.graph is None or not self.local_agents:
            # Dummy call for integration
            self.last_stats = {
                'total_reward': 0.0,
                'steps': 0,
                'final_cut_size': 0.0,
                'final_balance': 0.0,
                'final_conductance': 0.0,
                'epsilon': 1.0
            }
            return {
                'total_reward': 0.0,
                'steps': 1,
                'final_cut_size': 0.0,
                'final_balance': 0.0,
                'final_conductance': 0.0,
            }
        rewards = []
        cut_sizes = []
        balances = []
        conductances = []
        episode_metrics = []
        
        # Initialize visualizer
        visualizer = TrainingVisualizer(f'runs/{self.experiment_name}')
        
        # Create plots directory if it doesn't exist
        Path('plots').mkdir(exist_ok=True)
        
        # Training loop
        for episode in range(self.config.num_episodes):
            total_reward = 0.0
            steps = 0
            # For each agent (node), select partition
            for agent in self.local_agents:
                # Use get_state() if available, else fallback
                if hasattr(agent, 'get_state'):
                    state = agent.get_state()
                else:
                    # Construct a minimal AgentState
                    node_features = self.graph.get_node_features()[agent.node_id]
                    partition_sizes = np.array([len(p.nodes) for p in self.partitions])
                    partition_densities = np.array([getattr(p, 'density', 0.0) for p in self.partitions])
                    metrics_dict = {
                        'cut_size': compute_cut_size(self.graph, self.partitions),
                        'balance': compute_balance(self.partitions),
                        'conductance': compute_conductance(self.graph, self.partitions)
                    }
                    state = AgentState(
                        node_features=node_features,
                        partition_sizes=partition_sizes,
                        partition_densities=partition_densities,
                        graph_metrics=metrics_dict
                    )
                action, _ = agent.select_partition(state) if hasattr(agent, 'select_partition') else (np.random.randint(len(self.partitions)), 0)
                
                # Find current partition of the node
                current_partition = None
                for pid, p in self.graph.partitions.items():
                    if agent.node_id in p.nodes:
                        current_partition = pid
                        break
                
                # Get target partition ID
                target_partition = list(self.graph.partitions.keys())[action]
                
                # Move node to selected partition if it's different
                if current_partition != target_partition:
                    self.graph.move_node(agent.node_id, current_partition, target_partition)
                    # Update our local partition list
                    self.partitions = list(self.graph.partitions.values())
                # Compute reward (stub: negative cut size, to be improved)
                reward = -compute_cut_size(self.graph, self.partitions)
                total_reward += reward
                steps += 1
                rewards.append(total_reward)
                # Optionally: agent.memory.append(...), agent.learn(), etc.

            # --- Partition balancing/merging/splitting logic ---
            # Use graph's is_balanced method to check if balancing is needed
            if hasattr(self.graph, 'is_balanced') and hasattr(self.graph, 'balance_partitions'):
                if not self.graph.is_balanced():
                    self.graph.balance_partitions()
                    # Update self.partitions reference if needed
                    if hasattr(self.graph, 'partitions'):
                        # Convert dict to list if needed
                        if isinstance(self.graph.partitions, dict):
                            self.partitions = list(self.graph.partitions.values())
                        else:
                            self.partitions = self.graph.partitions
            # Example: merge small partitions (optional, can be improved)
            min_size = min(len(p.nodes) for p in self.partitions)
            if min_size < 2 and len(self.partitions) > 1:
                # Merge the smallest partition with the next
                small_partition = [p for p in self.partitions if len(p.nodes) == min_size][0]
                other_partition = [p for p in self.partitions if p.id != small_partition.id][0]
                # Move nodes from small partition to other partition
                for node in list(small_partition.nodes):  # Create a copy of nodes to avoid modification during iteration
                    small_partition.remove_node(node)
                    other_partition.add_node(node)
                # Remove small partition from list
                self.partitions.remove(small_partition)
            # Example: split large partitions (optional, can be improved)
            max_size = max(len(p.nodes) for p in self.partitions)
            target_size = self.graph.num_nodes // len(self.partitions)
            if max_size > 2 * target_size:
                # Find the largest partition
                large_partition = [p for p in self.partitions if len(p.nodes) == max_size][0]
                # Create a new partition
                new_partition = Partition(id=max(p.id for p in self.partitions) + 1)
                # Move half the nodes to the new partition
                nodes_to_move = list(large_partition.nodes)[:len(large_partition.nodes)//2]
                for node in nodes_to_move:
                    large_partition.remove_node(node)
                    new_partition.add_node(node)
                # Add new partition to list
                self.partitions.append(new_partition)
            # --- End partition management logic ---
            
            # Compute final metrics for this episode
            cut_size = compute_cut_size(self.graph, self.partitions)
            balance = compute_balance(self.partitions)
            conductance = compute_conductance(self.graph, self.partitions)
            
            # Store episode metrics
            episode_metrics.append({
                'reward': total_reward,
                'cut_size': cut_size,
                'balance': balance,
                'conductance': conductance,
                'epsilon': getattr(self.local_agents[0], 'epsilon', 1.0) if self.local_agents else 1.0
            })
            
            # Log metrics to TensorBoard
            visualizer.log_metrics({
                'reward': total_reward,
                'cut_size': cut_size,
                'balance': balance,
                'conductance': conductance,
                'epsilon': getattr(self.local_agents[0], 'epsilon', 1.0) if self.local_agents else 1.0
            }, episode)
            
            # Update metric lists for plotting
            rewards.append(total_reward)
            cut_sizes.append(cut_size)
            balances.append(balance)
            conductances.append(conductance)
            
            # Decay epsilon
            if self.local_agents:
                self.local_agents[0].epsilon *= self.config.epsilon_decay
        self.last_stats = {
            'total_reward': total_reward,
            'steps': steps,
            'final_cut_size': cut_size,
            'final_balance': balance,
            'final_conductance': conductance,
            'epsilon': getattr(self.local_agents[0], 'epsilon', 1.0) if self.local_agents else 1.0
        }
        return self.last_stats

    def get_partitions(self):
        """
        Return the current partitions as a dict mapping partition IDs to Partition objects.
        """
        return {p.id: p for p in self.partitions}

    def get_overhead_stats(self):
        return {}

    def partition(self, graph):
        """
        Partition the graph using RL-based dynamic partitioning.
        """
        from src.config.system_config import AgentConfig
        agent_config = getattr(self.config, 'agent_config', None)
        if agent_config is None:
            agent_config = AgentConfig()
        self.initialize(graph, agent_config)
        self.train()
        # Convert partitions to dict format
        result = {}
        for p in self.partitions:
            result[p.id] = set(p.nodes)
        return result



