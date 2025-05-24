# src/strategies/dynamic_partitioning.py

import numpy as np
import sys
import os
import time
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.local_agent import LocalAgent
from src.agents.base_agent import AgentState
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
        Enhanced RL training loop with validation-based early stopping and advanced curriculum learning.
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
        
        # Enhanced training tracking
        rewards = []
        cut_sizes = []
        balances = []
        conductances = []
        epsilons = []
        steps_per_episode = []
        episode_metrics = []
        validation_scores = []
        learning_rates = []
        
        # Validation-based early stopping
        validation_patience = 30
        validation_counter = 0
        best_validation_score = float('-inf')
        validation_threshold = 0.01  # Minimum improvement threshold
        
        # Initialize visualizer
        tb_dir = Path('runs') / f'{self.experiment_name}'
        visualizer = TrainingVisualizer(str(tb_dir))
        
        # Calculate total episodes (configured + potential additional)
        base_episodes = self.config.num_episodes
        total_episodes = base_episodes
        additional_iterations = 0
        
        # Get max_steps from config
        max_steps = getattr(self.config, 'max_steps', 100)
        
        # Set up progress logging
        logging.info(f"Starting enhanced training with {total_episodes} episodes")
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
        best_validation_episode = 0
        
        # Initialize tracking variables
        episode_timeouts = 0
        
        # Advanced curriculum learning phases
        curriculum_phases = [
            {
                'name': 'Foundation',
                'episodes': int(total_episodes * 0.2),
                'balance_weight': 80.0,
                'cut_weight': 10.0,
                'conductance_weight': 5.0,
                'exploration_bonus': 2.0
            },
            {
                'name': 'Development', 
                'episodes': int(total_episodes * 0.3),
                'balance_weight': 60.0,
                'cut_weight': 30.0,
                'conductance_weight': 15.0,
                'exploration_bonus': 1.0
            },
            {
                'name': 'Refinement',
                'episodes': int(total_episodes * 0.3),
                'balance_weight': 40.0,
                'cut_weight': 45.0,
                'conductance_weight': 25.0,
                'exploration_bonus': 0.5
            },
            {
                'name': 'Optimization',
                'episodes': total_episodes,  # Rest of episodes
                'balance_weight': 25.0,
                'cut_weight': 55.0,
                'conductance_weight': 35.0,
                'exploration_bonus': 0.0
            }
        ]
        
        current_phase = 0
        
        # Training loop
        for episode in range(total_episodes):
            episode_start_time = time.time()
            total_reward = 0.0
            steps = 0
            
            # Determine current curriculum phase
            while (current_phase < len(curriculum_phases) - 1 and 
                   episode >= curriculum_phases[current_phase]['episodes']):
                current_phase += 1
                logging.info(f"Advancing to curriculum phase: {curriculum_phases[current_phase]['name']}")
            
            phase = curriculum_phases[current_phase]
            
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
                    
                    partition_sizes = torch.tensor([len(p.nodes) for p in self.partitions], dtype=torch.float32)
                    partition_densities = torch.tensor([getattr(p, 'density', 0.0) for p in self.partitions], dtype=torch.float32)
                    metrics_dict = {
                        'cut_size': float(compute_cut_size(self.graph, self.partitions)),
                        'balance': float(compute_balance(self.partitions)),
                        'conductance': float(compute_conductance(self.graph, self.partitions))
                    }
                    state = AgentState(
                        node_features=node_features,
                        partition_sizes=partition_sizes,
                        partition_densities=partition_densities,
                        graph_metrics=metrics_dict
                    )
                
                # Store the current state
                previous_states[agent.node_id] = state
                
                # Select action with enhanced exploration in early phases
                if hasattr(agent, 'select_partition'):
                    action, _ = agent.select_partition(state)
                else:
                    # Enhanced epsilon for exploration bonus
                    enhanced_epsilon = min(1.0, agent.epsilon + phase['exploration_bonus'] * (1.0 - episode / total_episodes))
                    action, _ = agent.select_action(state, enhanced_epsilon)
                
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
            
            # Track steps per episode properly
            steps_per_episode.append(steps)
            
            # Calculate target partition size
            target_size = self.graph.num_nodes / len(self.partitions)
            
            # Enhanced reward calculation with curriculum phase weights
            cut_improvement = (prev_cut_size - cut_size) / (prev_cut_size + 1e-8)
            balance_improvement = balance - prev_balance
            conductance_improvement = (prev_conductance - conductance) / (prev_conductance + 1e-8)
            
            # Get curriculum weights
            balance_weight = phase['balance_weight']
            cut_weight = phase['cut_weight']
            conductance_weight = phase['conductance_weight']
            
            # Normalize the improvements to prevent large value swings
            norm_cut_imp = cut_improvement / (prev_cut_size + 1e-8)
            norm_balance_imp = balance_improvement
            norm_cond_imp = conductance_improvement / (prev_conductance + 1e-8)
            
            # Calculate enhanced penalties
            target_size = self.graph.num_nodes / len(self.partitions)
            max_size_diff = max(abs(len(p.nodes) - target_size) for p in self.partitions)
            size_variance_penalty = max_size_diff / target_size
            
            # Adaptive penalty scaling based on phase
            penalty_scale = 1.0 + (current_phase * 0.5)  # Increase penalties as training progresses
            
            # Enhanced reward function with curriculum learning weights and adaptive penalties
            reward = (
                cut_weight * norm_cut_imp +                           # Dynamic cut weight
                balance_weight * norm_balance_imp +                   # Dynamic balance weight  
                conductance_weight * norm_cond_imp +                  # Dynamic conductance weight
                -30.0 * penalty_scale * (1 - balance) +              # Adaptive imbalance penalty
                -15.0 * penalty_scale * size_variance_penalty +      # Adaptive size variance penalty
                -5.0 * (cut_size / (self.graph.num_nodes * 2)) +     # Normalized absolute cut penalty
                phase['exploration_bonus'] * 0.1                      # Small exploration bonus
            )
            
            # Clip reward to prevent extreme values
            reward = np.clip(reward, -15.0, 15.0)  # Slightly wider range for enhanced rewards
            
            # Store this episode's metrics to compare with next episode
            prev_cut_size = cut_size
            prev_balance = balance
            prev_conductance = conductance
            
            # Enhanced state calculation and agent training
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
                    
                    partition_sizes = torch.tensor([len(p.nodes) for p in self.partitions], dtype=torch.float32)
                    partition_densities = torch.tensor([getattr(p, 'density', 0.0) for p in self.partitions], dtype=torch.float32)
                    metrics_dict = {
                        'cut_size': float(cut_size),
                        'balance': float(balance),
                        'conductance': float(conductance)
                    }
                    next_state = AgentState(
                        node_features=node_features,
                        partition_sizes=partition_sizes,
                        partition_densities=partition_densities,
                        graph_metrics=metrics_dict
                    )
                
                # Store transition with enhanced individual reward calculation
                current_p = None
                for pid, p in self.graph.partitions.items():
                    if agent.node_id in p.nodes:
                        current_p = pid
                        break
                
                action_idx = list(self.graph.partitions.keys()).index(current_p)
                
                # Enhanced individual reward calculation
                partition = next(p for p in self.partitions if agent.node_id in p.nodes)
                local_balance = len(partition.nodes) / target_size
                neighbor_cut = sum(1 for neighbor in self.graph.get_neighbors(agent.node_id)
                                if neighbor not in partition.nodes)
                neighbor_count = len(self.graph.get_neighbors(agent.node_id))
                
                # Enhanced individual reward with phase-specific bonuses
                individual_reward = (
                    reward +  # Global component
                    -5.0 * penalty_scale * abs(1 - local_balance) +  # Adaptive local balance penalty
                    -2.0 * (neighbor_cut / max(neighbor_count, 1)) +  # Local cut penalty (avoid division by zero)
                    phase['exploration_bonus'] * 0.05  # Small individual exploration bonus
                ) / 2.0  # Scale down combined reward
                
                # Clip individual reward
                individual_reward = np.clip(individual_reward, -15.0, 15.0)
                
                if hasattr(agent, 'store_transition'):
                    # Store the transition in the agent's memory
                    agent.store_transition(
                        previous_states[agent.node_id],  # Previous state
                        action_idx,                      # Action taken
                        individual_reward,               # Enhanced reward
                        next_state,                      # New state
                        False                            # Not terminal state
                    )
            
            # Enhanced training with validation monitoring
            if episode > 20 and episode % 3 == 0:  # Train every 3 episodes after longer warmup
                training_metrics = []
                for agent in self.local_agents:
                    if hasattr(agent, 'train_step'):
                        metrics = agent.train_step()
                        training_metrics.append(metrics)
                
                # Aggregate training metrics
                if training_metrics:
                    avg_loss = np.mean([m.get('loss', 0) for m in training_metrics])
                    avg_lr = np.mean([m.get('learning_rate', 0) for m in training_metrics])
                    avg_validation = np.mean([m.get('validation_score', 0) for m in training_metrics])
                    
                    learning_rates.append(avg_lr)
                    validation_scores.append(avg_validation)
                    
                    # Check for validation-based early stopping
                    if avg_validation > best_validation_score + validation_threshold:
                        best_validation_score = avg_validation
                        best_validation_episode = episode
                        validation_counter = 0
                        # Save best model state
                        best_partitions = self.get_partitions()
                        best_cut_size = cut_size
                    else:
                        validation_counter += 1
                
                    # Log enhanced training metrics
                    visualizer.log_metrics({
                        'training_loss': avg_loss,
                        'learning_rate': avg_lr,
                        'validation_score': avg_validation,
                        'curriculum_phase': current_phase
                    }, episode)
            
            total_reward += reward
            rewards.append(total_reward)
            
            # Enhanced partition management logic
            current_balance = compute_balance(self.partitions)
            if current_balance < 0.6:  # More aggressive threshold in later phases
                # Use graph's balancing methods if available
                if hasattr(self.graph, 'is_balanced') and hasattr(self.graph, 'balance_partitions'):
                    if not self.graph.is_balanced():
                        self.graph.balance_partitions()
                        if hasattr(self.graph, 'partitions'):
                            if isinstance(self.graph.partitions, dict):
                                self.partitions = list(self.graph.partitions.values())
                            else:
                                self.partitions = self.graph.partitions
            
            # Store metrics in their respective lists for tracking over time
            cut_sizes.append(cut_size)
            balances.append(balance)
            conductances.append(conductance)
            current_epsilon = self.local_agents[0].epsilon if self.local_agents else 1.0
            epsilons.append(current_epsilon)
            
            # Store enhanced episode metrics
            episode_metrics.append({
                'reward': total_reward,
                'cut_size': cut_size,
                'balance': balance,
                'conductance': conductance,
                'epsilon': current_epsilon,
                'curriculum_phase': current_phase,
                'phase_name': phase['name']
            })
            
            # Enhanced early stopping logic
            if episode >= 60:  # Longer initial learning period
                lookback_window = 40  # Larger window for stability
                recent_cut_sizes_es = cut_sizes[-lookback_window:]
                recent_rewards_es = rewards[-lookback_window:]
                
                # Calculate coefficient of variation for stability check
                cut_cv = np.std(recent_cut_sizes_es) / (np.mean(recent_cut_sizes_es) + 1e-8)
                reward_cv = np.std(recent_rewards_es) / (abs(np.mean(recent_rewards_es)) + 1e-8)
                
                # Check improvement rate
                if len(cut_sizes) >= 80:
                    first_half = np.mean(cut_sizes[-80:-40])
                    second_half = np.mean(recent_cut_sizes_es)
                    cut_improvement_rate = abs(first_half - second_half) / (first_half + 1e-8)
                    
                    # Enhanced early stopping criteria
                    convergence_criteria = (
                        cut_cv < 0.015 and          # Cut size very stable (< 1.5% variation)
                        reward_cv < 0.08 and        # Rewards very stable (< 8% variation)
                        cut_improvement_rate < 0.003 and  # Very small improvement (< 0.3%)
                        balance >= 0.75             # Good balance achieved
                    )
                    
                    validation_criteria = (
                        validation_counter >= validation_patience and
                        episode - best_validation_episode > validation_patience
                    )
                    
                    if convergence_criteria or validation_criteria:
                        stop_reason = "convergence" if convergence_criteria else "validation plateau"
                        logging.info(f"Enhanced early stopping at episode {episode+1} due to {stop_reason}: "
                                   f"Cut CV: {cut_cv:.4f}, Reward CV: {reward_cv:.4f}, "
                                   f"Improvement: {cut_improvement_rate:.4f}, Validation counter: {validation_counter}")
                        break
                        
            # Check for episode timeout
            if steps >= max_steps:
                episode_timeouts += 1
                if episode_timeouts > total_episodes * 0.7:  # Lower threshold for timeout warning
                    logging.warning(f"High timeout rate detected ({episode_timeouts}/{episode+1}). "
                                  f"Consider increasing max_steps or improving reward function.")
            
            # Enhanced logging to TensorBoard
            visualizer.log_metrics({
                'reward': float(total_reward),
                'cut_size': float(cut_size),
                'balance': float(balance),
                'conductance': float(conductance),
                'epsilon': float(current_epsilon),
                'curriculum_phase': current_phase,
                'steps_per_episode': steps
            }, episode)
            
            # Enhanced progress logging
            if self.track_learning and (episode + 1) % self.log_interval == 0:
                window = min(self.window_size, episode + 1)
                recent_cut_sizes = cut_sizes[-window:]
                recent_balances = balances[-window:]
                recent_conductances = conductances[-window:]
                
                if episode >= 30:
                    first_30_cut = np.mean(cut_sizes[:30])
                    recent_cut = np.mean(recent_cut_sizes)
                    cut_improvement = ((first_30_cut - recent_cut) / first_30_cut) * 100
                    
                    cut_arrow = "↓" if cut_improvement > 0 else "↑"
                    cut_improvement_abs = abs(cut_improvement)
                    
                    episode_time = time.time() - episode_start_time
                    elapsed_time = time.time() - start_time
                    estimated_remaining = (elapsed_time / (episode + 1)) * (total_episodes - episode - 1)
                    
                    current_lr = learning_rates[-1] if learning_rates else self.local_agents[0].optimizer.param_groups[0]['lr'] if self.local_agents else 0.001
                    
                    logging.info(f"Episode {episode+1}/{total_episodes} "
                               f"[{elapsed_time:.1f}s elapsed, ~{estimated_remaining:.1f}s remaining] - "
                               f"Phase: {phase['name']}, "
                               f"Cut: {np.mean(recent_cut_sizes):.2f} "
                               f"({cut_arrow}{cut_improvement_abs:.1f}%), "
                               f"Balance: {np.mean(recent_balances):.4f}, "
                               f"Conductance: {np.mean(recent_conductances):.4f}, "
                               f"ε: {current_epsilon:.4f}, LR: {current_lr:.6f}")
                else:
                    logging.info(f"Episode {episode+1}/{total_episodes} - "
                               f"Phase: {phase['name']}, "
                               f"Cut: {np.mean(recent_cut_sizes):.2f}, "
                               f"Balance: {np.mean(recent_balances):.4f}, "
                               f"Conductance: {np.mean(recent_conductances):.4f}, "
                               f"ε: {current_epsilon:.4f}")
        
        # Log final time and results
        elapsed_time = time.time() - start_time
        logging.info(f"Enhanced training completed in {elapsed_time:.2f}s")
        logging.info(f"Final curriculum phase: {curriculum_phases[current_phase]['name']}")
        if validation_scores:
            logging.info(f"Best validation score: {best_validation_score:.4f} at episode {best_validation_episode}")
        
        # Ensure visualizer is properly closed
        try:
            if 'visualizer' in locals() and visualizer:
                visualizer.close()
                logging.debug(f"Enhanced training visualizer for {self.experiment_name} closed successfully")
        except Exception as e:
            logging.warning(f"Error closing enhanced visualizer: {e}")
        
        # Return enhanced training statistics
        return {
            'rewards': rewards,
            'cut_sizes': cut_sizes,
            'balances': balances,
            'conductances': conductances,
            'epsilons': epsilons,
            'steps': steps_per_episode,
            'episode_metrics': episode_metrics,
            'validation_scores': validation_scores,
            'learning_rates': learning_rates,
            'best_cut_size': best_cut_size,
            'best_partitions': best_partitions,
            'training_time': elapsed_time,
            'curriculum_phases': [p['name'] for p in curriculum_phases],
            'final_phase': curriculum_phases[current_phase]['name'],
            'best_validation_score': best_validation_score,
            'best_validation_episode': best_validation_episode
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



