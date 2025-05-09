# src/strategies/dynamic_partitioning.py

import numpy as np
import sys
import os
import time
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.local_agent import *
from src.config.system_config import *
from src.core.graph import *
from pathlib import Path
from src.utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance
from src.utils.visualization import TrainingVisualizer

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
        self.total_episodes = getattr(config, 'num_episodes', 100)  # Set default value if not in config
        
        # Initialize progress tracking
        self.learning_progress = {
            'rewards': [],
            'cut_sizes': [],
            'balances': [],
            'conductances': [],
            'epsilons': []
        }
        
        # Learning progress tracking settings
        self.track_learning = getattr(config, 'track_learning_progress', True)
        self.log_interval = getattr(config, 'learning_log_interval', 100)
        self.window_size = getattr(config, 'rolling_window_size', 20)

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
        
        # Log initial graph state
        initial_cut_size = compute_cut_size(self.graph, self.partitions)
        initial_balance = compute_balance(self.partitions)
        initial_conductance = compute_conductance(self.graph, self.partitions)
        
        logging.info(f"Initial partitioning metrics - Cut size: {initial_cut_size:.4f}, "
                    f"Balance: {initial_balance:.4f}, Conductance: {initial_conductance:.4f}")

    def train(self, continue_iterations=False, max_additional_iterations=10):
        """
        RL training loop: each agent selects a partition, environment updates, reward assigned.
        Agents learn from their experiences through memory replay.
        Returns stats for integration/testing.
        
        Parameters:
        -----------
        continue_iterations: bool
            If True, after completing the configured number of episodes, will prompt 
            the user to continue with more iterations.
        max_additional_iterations: int
            Maximum number of additional iterations beyond the configured episodes
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
        epsilons = []
        episode_metrics = []
        
        # Initialize visualizer
        visualizer = TrainingVisualizer(f'runs/{self.experiment_name}')
        
        # Calculate total episodes (configured + potential additional)
        base_episodes = self.config.num_episodes
        total_episodes = base_episodes
        additional_iterations = 0
        
        # Set up progress logging
        logging.info(f"Starting training with {total_episodes} episodes")
        start_time = time.time()
        
        # Get initial partition state
        initial_cut_size = compute_cut_size(self.graph, self.partitions)
        initial_balance = compute_balance(self.partitions)
        initial_conductance = compute_conductance(self.graph, self.partitions)
        
        # Track metrics from previous episode for calculating improvements
        prev_cut_size = initial_cut_size
        prev_balance = initial_balance
        prev_conductance = initial_conductance
        
        # Save initial partition state to ensure we can properly track progress
        best_partitions = self.get_partitions()
        best_cut_size = initial_cut_size
        
        # Training loop
        for episode in range(total_episodes):
            episode_start_time = time.time()
            total_reward = 0.0
            steps = 0
            
            # Store agent states for each node before actions
            previous_states = {}
            
            # For each agent (node), select partition
            for agent in self.local_agents:
                # Get current state
                if hasattr(agent, 'get_state'):
                    state = agent.get_state()
                else:
                    # Construct a minimal AgentState
                    node_features = self.graph.get_node_features()[agent.node_id]
                    
                    # Handle potential dimension mismatch between config and actual features
                    expected_dim = getattr(agent.config, 'feature_dim', 24)
                    actual_dim = len(node_features)
                    
                    if actual_dim != expected_dim:
                        # Resize features to match expected dimension
                        if actual_dim < expected_dim:
                            # Pad with zeros to reach expected dimension
                            logging.debug(f"Padding node features from {actual_dim} to {expected_dim} dimensions")
                            padding = torch.zeros(expected_dim - actual_dim)
                            node_features = torch.cat([node_features, padding])
                        else:
                            # Truncate to expected dimension
                            logging.debug(f"Truncating node features from {actual_dim} to {expected_dim} dimensions")
                            node_features = node_features[:expected_dim]
                    
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
                
                # Store the current state
                previous_states[agent.node_id] = state
                
                # Select action
                action, _ = agent.select_partition(state) if hasattr(agent, 'select_partition') else agent.select_action(state, agent.epsilon)[0]
                
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
                    
                steps += 1
            
            # Calculate metrics after all agents have taken actions
            cut_size = compute_cut_size(self.graph, self.partitions)
            balance = compute_balance(self.partitions)
            conductance = compute_conductance(self.graph, self.partitions)
            
            # Calculate improvements from previous state (negative value is better for cut size and conductance)
            cut_improvement = prev_cut_size - cut_size  
            balance_improvement = balance - prev_balance  # positive value is better for balance
            conductance_improvement = prev_conductance - conductance
            
            # Create a reward that emphasizes improvement rather than absolute values
            # This is critical for learning - reward improvements, penalize degradation
            reward = 10.0 * cut_improvement + 20.0 * balance_improvement + 5.0 * conductance_improvement
            
            # Store this episode's metrics to compare with next episode
            prev_cut_size = cut_size
            prev_balance = balance
            prev_conductance = conductance
            
            # Get the new state for each agent after all moves have been made
            for agent in self.local_agents:
                # Get new state after all actions
                if hasattr(agent, 'get_state'):
                    next_state = agent.get_state()
                else:
                    node_features = self.graph.get_node_features()[agent.node_id]
                    
                    # Handle dimension mismatch
                    expected_dim = getattr(agent.config, 'feature_dim', 24)
                    actual_dim = len(node_features)
                    
                    if actual_dim != expected_dim:
                        if actual_dim < expected_dim:
                            padding = torch.zeros(expected_dim - actual_dim)
                            node_features = torch.cat([node_features, padding])
                        else:
                            node_features = node_features[:expected_dim]
                    
                    partition_sizes = np.array([len(p.nodes) for p in self.partitions])
                    partition_densities = np.array([getattr(p, 'density', 0.0) for p in self.partitions])
                    metrics_dict = {
                        'cut_size': cut_size,
                        'balance': balance,
                        'conductance': conductance
                    }
                    next_state = AgentState(
                        node_features=node_features,
                        partition_sizes=partition_sizes,
                        partition_densities=partition_densities,
                        graph_metrics=metrics_dict
                    )
                
                # Store transition in agent's memory with correct action and INDIVIDUAL reward
                # Find which partition the agent is in now
                current_p = None
                for pid, p in self.graph.partitions.items():
                    if agent.node_id in p.nodes:
                        current_p = pid
                        break
                
                # Get the index of the action that was taken
                action_idx = list(self.graph.partitions.keys()).index(current_p)
                
                # Calculate individual reward based on how this node's neighbors are distributed
                individual_reward = reward
                # Add extra reward if node is in a balanced partition
                if hasattr(agent, 'store_transition'):
                    # Store the transition in the agent's memory
                    agent.store_transition(
                        previous_states[agent.node_id],  # Previous state
                        action_idx,                      # Action taken
                        individual_reward,               # Reward
                        next_state,                      # New state
                        False                            # Not terminal state
                    )
            
            # Perform learning step after all agents have stored their transitions
            # This ensures we have enough data for effective batch learning
            if episode > 10 and episode % 5 == 0:  # Train every 5 episodes after warmup
                for agent in self.local_agents:
                    if hasattr(agent, 'train_step'):
                        agent.train_step()
            
            total_reward += reward
            rewards.append(total_reward)
            
            # --- Partition balancing/merging/splitting logic ---
            # This section remains mostly unchanged but could be improved
            current_balance = compute_balance(self.partitions)
            if current_balance < 0.7:
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
                
                # Only perform partition merging if there are extremely small partitions
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
                
                # Only perform partition splitting if there are extremely large partitions
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
            
            # Track best partitioning solution
            if cut_size < best_cut_size and balance >= 0.7:
                best_cut_size = cut_size
                best_partitions = self.get_partitions()
            
            # Store metrics in their respective lists for tracking over time
            cut_sizes.append(cut_size)
            balances.append(balance)
            conductances.append(conductance)
            current_epsilon = getattr(self.local_agents[0], 'epsilon', 1.0) if self.local_agents else 1.0
            epsilons.append(current_epsilon)
            
            # Store episode metrics
            episode_metrics.append({
                'reward': total_reward,
                'cut_size': cut_size,
                'balance': balance,
                'conductance': conductance,
                'epsilon': current_epsilon
            })
            
            # Log metrics to TensorBoard
            visualizer.log_metrics({
                'reward': total_reward,
                'cut_size': cut_size,
                'balance': balance,
                'conductance': conductance,
                'epsilon': current_epsilon
            }, episode)
            
            # Rest of the training loop remains mostly unchanged
            # Log progress at intervals
            if self.track_learning and (episode + 1) % self.log_interval == 0:
                # Calculate rolling averages for smoother trend reporting
                window = min(self.window_size, episode + 1)
                recent_cut_sizes = cut_sizes[-window:]
                recent_balances = balances[-window:]
                recent_conductances = conductances[-window:]
                
                # Calculate improvements from start (if we have enough episodes)
                if episode >= 20:
                    first_20_cut = np.mean(cut_sizes[:20])
                    recent_cut = np.mean(recent_cut_sizes)
                    cut_improvement = ((first_20_cut - recent_cut) / first_20_cut) * 100
                    
                    # Determine arrow direction based on whether improvement is positive or negative
                    cut_arrow = "↓" if cut_improvement > 0 else "↑"
                    cut_improvement_abs = abs(cut_improvement)
                    
                    episode_time = time.time() - episode_start_time
                    elapsed_time = time.time() - start_time
                    estimated_remaining = (elapsed_time / (episode + 1)) * (total_episodes - episode - 1)
                    
                    logging.info(f"Episode {episode+1}/{total_episodes} "
                               f"[{elapsed_time:.1f}s elapsed, ~{estimated_remaining:.1f}s remaining] - "
                               f"Cut: {np.mean(recent_cut_sizes):.2f} "
                               f"({cut_arrow}{cut_improvement_abs:.1f}%), "
                               f"Balance: {np.mean(recent_balances):.4f}, "
                               f"Conductance: {np.mean(recent_conductances):.4f}, "
                               f"ε: {current_epsilon:.4f}")
                else:
                    logging.info(f"Episode {episode+1}/{total_episodes} - "
                                f"Cut: {np.mean(recent_cut_sizes):.2f}, "
                                f"Balance: {np.mean(recent_balances):.4f}, "
                                f"Conductance: {np.mean(recent_conductances):.4f}, "
                                f"ε: {current_epsilon:.4f}")
            
        # Log final time
        elapsed_time = time.time() - start_time
        logging.info(f"Training completed in {elapsed_time:.2f}s")
        
        # Ensure visualizer is properly closed to prevent hanging on program exit
        try:
            if 'visualizer' in locals() and visualizer:
                visualizer.close()
                logging.debug(f"Training visualizer for {self.experiment_name} closed successfully")
        except Exception as e:
            logging.warning(f"Error closing visualizer: {e}")
        
        # Return training statistics
        return {
            'rewards': rewards,
            'cut_sizes': cut_sizes,
            'balances': balances,
            'conductances': conductances,
            'epsilons': epsilons,
            'episode_metrics': episode_metrics,
            'best_cut_size': best_cut_size,
            'best_partitions': best_partitions,
            'training_time': elapsed_time
        }

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
        stats = self.train()
        # Convert partitions to dict format
        result = {}
        for p in self.partitions:
            result[p.id] = set(p.nodes)
        return result



