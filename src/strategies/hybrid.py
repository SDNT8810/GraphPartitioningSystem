"""
HybridPartitioningStrategy: Combines spectral and RL-based partitioning.
Applies robust partition management (        # Convert spectral partitions to graph.partitions format
        graph.partitions = {}
        for pid, nodes in partitions.items():
            graph.partitions[pid] = Partition(id=pid, nodes=nodes)
            
        # Initialize RL with current partitions
        self.rl.initialize(graph, agent_config)
        
        # Train the RL agents
        rl_start = time.time()
        rl_stats = self.rl.train()
        rl_time = time.time() - rl_start
        
        # Get partitions after RL training
        rl_partitions = {}
        for p in self.rl.partitions:
            if hasattr(p, 'nodes'):
                rl_partitions[p.id] = set(p.nodes)
            else:
                rl_partitions[p.id] = set()
        
        # Calculate RL metrics
        rl_cut_size = compute_cut_size(graph, rl_partitions), splitting) after assignment.

See also:
- [CODEMAProposed_Method](../../CODEMAProposed_Method)
- [TODO.md](../../TODO.md)
- [INDEX.md](../INDEX.md)
"""

from typing import Dict, Set, List, Any
import numpy as np
import time
import logging
import torch
from pathlib import Path
from .spectral import SpectralPartitioningStrategy
from .dynamic_partitioning import DynamicPartitioning
from ..core.graph import Graph, Partition
from ..utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance
from ..utils.visualization import *

class HybridPartitioningStrategy:
    """
    Hybrid partitioning: spectral initialization, RL refinement, robust partition management.
    """
    def __init__(self, config, experiment_name: str = "hybrid"):
        self.config = config
        self.experiment_name = experiment_name
        self.spectral = SpectralPartitioningStrategy(config)
        self.rl = DynamicPartitioning(config, experiment_name=experiment_name)
        
        # Initialize metrics tracking
        self.metrics_history = {
            'spectral': {
                'cut_sizes': [],
                'balances': [],
                'conductances': []
            },
            'dynamic': {
                'cut_sizes': [],
                'balances': [],
                'conductances': []
            },
            'hybrid': {
                'cut_sizes': [],
                'balances': [],
                'conductances': []
            }
        }
        
        # Learning progress tracking settings
        self.track_learning = getattr(config, 'track_learning_progress', True)
        self.log_interval = getattr(config, 'learning_log_interval', 100)

    def partition(self, graph: Graph, run_id: int = 0) -> Dict[int, Set]:
        """
        1. Use spectral partitioning to initialize.
        2. Optionally refine with RL-based partitioning.
        3. Apply balancing, merging, and splitting as in other strategies.
        
        Args:
            graph: Input graph object
            run_id: Run identifier (used for creating unique directories)
        """
        # Store run_id in environment for other methods to access
        import os
        os.environ['HYBRID_RUN_ID'] = str(run_id)
        
        start_time = time.time()
        logging.info(f"Starting hybrid partitioning for {self.experiment_name} (run {run_id})")
        
        # Create a unique directory for each run to avoid conflicts
        tb_dir = Path('runs').joinpath(f'{self.experiment_name}_run{run_id}')

        Path(tb_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize visualizer
        visualizer = TrainingVisualizer(tb_dir)
        
        # Pass config to graph
        graph.config = self.config
        
        # Step 1: Spectral initialization
        logging.info("Phase 1: Applying spectral initialization...")
        spectral_start = time.time()
        partitions = self.spectral.partition(graph)
        spectral_time = time.time() - spectral_start
        
        # Calculate spectral metrics
        spectral_cut_size = compute_cut_size(graph, partitions)
        spectral_balance = compute_balance(partitions)
        spectral_conductance = compute_conductance(graph, partitions)
        
        # Store metrics
        self.metrics_history['spectral']['cut_sizes'].append(spectral_cut_size)
        self.metrics_history['spectral']['balances'].append(spectral_balance)
        self.metrics_history['spectral']['conductances'].append(spectral_conductance)
        
        # Log spectral results
        logging.info(f"Spectral partitioning completed in {spectral_time:.2f}s")
        logging.info(f"Spectral metrics - Cut size: {spectral_cut_size:.4f}, " 
                    f"Balance: {spectral_balance:.4f}, Conductance: {spectral_conductance:.4f}")
        
        # Log to TensorBoard
        visualizer.log_metrics({
            'spectral/cut_size': spectral_cut_size,
            'spectral/balance': spectral_balance,
            'spectral/conductance': spectral_conductance,
            'spectral/time': spectral_time
        }, step=0)
        
        # Step 2: RL refinement
        logging.info("Phase 2: Applying RL-based refinement...")
        from src.config.system_config import AgentConfig
        agent_config = getattr(self.config, 'agent_config', None)
        if agent_config is None:
            agent_config = AgentConfig()
            
        # Convert spectral partitions to graph.partitions format properly
        graph.partitions = {}
        for pid, nodes in partitions.items():
            # Create a Partition object with proper nodes (as a list or set, not a Partition)
            partition = Partition(id=pid)
            
            # Convert nodes to a list if it's not already iterable
            if isinstance(nodes, set) or isinstance(nodes, list):
                for node in nodes:
                    # Make sure we're adding integers, not other Partition objects
                    if isinstance(node, int):
                        partition.add_node(node)
                    elif hasattr(node, 'id'):  # If it's another Partition object
                        logging.warning(f"Found Partition object instead of node ID: {node}")
                        continue
                    else:
                        try:
                            node_id = int(node)  # Try to convert to int
                            partition.add_node(node_id)
                        except (ValueError, TypeError):
                            logging.warning(f"Skipping invalid node: {node}")
            else:
                # Handle single node case
                try:
                    if isinstance(nodes, int):
                        partition.add_node(nodes)
                    else:
                        node_id = int(nodes)
                        partition.add_node(node_id)
                except (ValueError, TypeError):
                    logging.warning(f"Skipping invalid node: {nodes}")
                
            graph.partitions[pid] = partition
            
        # Make a clean copy of the graph to pass to the RL component
        import copy
        graph_copy = copy.deepcopy(graph)
        
        # Initialize RL with current partitions
        try:
            # Debug output
            logging.info(f"Initializing RL with {len(graph_copy.partitions)} partitions")
            for pid, partition in graph_copy.partitions.items():
                if hasattr(partition, 'nodes') and hasattr(partition.nodes, '__len__'):
                    logging.info(f"  Partition {pid}: {len(partition.nodes)} nodes")
                else:
                    logging.info(f"  Partition {pid}: invalid nodes attribute")
            
            self.rl.initialize(graph_copy, agent_config)
            
            # Train the RL agents
            rl_start = time.time()
            rl_stats = self.rl.train()
            rl_time = time.time() - rl_start
            
            # Get partitions after RL training safely
            rl_partitions = {}
            for p in self.rl.partitions:
                if hasattr(p, 'id') and hasattr(p, 'nodes'):
                    # Handle potential non-iterable nodes attribute
                    try:
                        rl_partitions[p.id] = set(p.nodes)
                    except TypeError:
                        # If nodes is not iterable, create a singleton set
                        rl_partitions[p.id] = {p.nodes}
        except Exception as e:
            logging.error(f"Error during RL training: {e}")
            # Fall back to spectral partitions
            logging.info("Falling back to spectral partitioning due to RL error")
            rl_partitions = partitions
            rl_time = 0.0
        
        # Calculate RL metrics
        rl_cut_size = compute_cut_size(graph, rl_partitions)
        rl_balance = compute_balance(rl_partitions)
        rl_conductance = compute_conductance(graph, rl_partitions)
        
        # Store metrics
        self.metrics_history['dynamic']['cut_sizes'].append(rl_cut_size)
        self.metrics_history['dynamic']['balances'].append(rl_balance)
        self.metrics_history['dynamic']['conductances'].append(rl_conductance)
        
        # Log RL results
        logging.info(f"RL refinement completed in {rl_time:.2f}s")
        logging.info(f"RL metrics - Cut size: {rl_cut_size:.4f}, " 
                    f"Balance: {rl_balance:.4f}, Conductance: {rl_conductance:.4f}")
        
        # Log to TensorBoard
        visualizer.log_metrics({
            'rl/cut_size': rl_cut_size,
            'rl/balance': rl_balance,
            'rl/conductance': rl_conductance,
            'rl/time': rl_time
        }, step=0)
        
        # Update graph.partitions with RL results
        graph.partitions = {}
        for pid, nodes in rl_partitions.items():
            graph.partitions[pid] = Partition(id=pid, nodes=nodes)
            
        # Step 3: Robust partition management
        logging.info("Phase 3: Applying robust partition management...")
        robust_start = time.time()
        
        # Balance partitions if needed
        if hasattr(graph, 'is_balanced') and hasattr(graph, 'balance_partitions'):
            if not graph.is_balanced():
                logging.info("Balancing partitions...")
                graph.balance_partitions()
                
        # Merge small partitions
        min_size = min(len(p.nodes) for p in graph.partitions.values())
        if min_size < 2 and len(graph.partitions) > 1 and hasattr(graph, 'merge_partitions'):
            small_pid = [p.id for p in graph.partitions.values() if len(p.nodes) == min_size][0]
            other_pid = next((pid for pid in graph.partitions.keys() if pid != small_pid), None)
            if other_pid is not None:
                logging.info(f"Merging small partition {small_pid} into {other_pid}...")
                graph.merge_partitions(small_pid, other_pid)
                
        # Split large partitions
        if len(graph.partitions) > 0:
            max_size = max(len(p.nodes) for p in graph.partitions.values())
            avg_size = graph.num_nodes / len(graph.partitions)
            if max_size > 2 * avg_size and hasattr(graph, 'split_partition'):
                large_pid = [p.id for p in graph.partitions.values() if len(p.nodes) == max_size][0]
                logging.info(f"Splitting large partition {large_pid}...")
                graph.split_partition(large_pid)
                
        robust_time = time.time() - robust_start
        
        # Get final partitions
        final_partitions = {p.id: set(p.nodes) for p in graph.partitions.values()}
        
        # Calculate final metrics
        final_cut_size = compute_cut_size(graph, final_partitions)
        final_balance = compute_balance(final_partitions)
        final_conductance = compute_conductance(graph, final_partitions)
        
        # Store metrics
        self.metrics_history['hybrid']['cut_sizes'].append(final_cut_size)
        self.metrics_history['hybrid']['balances'].append(final_balance)
        self.metrics_history['hybrid']['conductances'].append(final_conductance)
        
        # Log final results
        total_time = time.time() - start_time
        logging.info(f"Robust partition management completed in {robust_time:.2f}s")
        logging.info(f"Final hybrid metrics - Cut size: {final_cut_size:.4f}, " 
                    f"Balance: {final_balance:.4f}, Conductance: {final_conductance:.4f}")
        logging.info(f"Total hybrid partitioning time: {total_time:.2f}s")
        
        # Log to TensorBoard
        visualizer.log_metrics({
            'hybrid/cut_size': final_cut_size,
            'hybrid/balance': final_balance,
            'hybrid/conductance': final_conductance,
            'hybrid/time': total_time
        }, step=0)
        
        # Calculate improvements
        cut_improvement = ((spectral_cut_size - final_cut_size) / spectral_cut_size) * 100
        balance_improvement = ((final_balance - spectral_balance) / spectral_balance) * 100
        conductance_improvement = ((spectral_conductance - final_conductance) / spectral_conductance) * 100
        
        # Log improvements
        logging.info(f"Improvements from spectral to hybrid:")
        logging.info(f"  Cut size: {cut_improvement:.1f}%")
        logging.info(f"  Balance: {balance_improvement:.1f}%")
        logging.info(f"  Conductance: {conductance_improvement:.1f}%")
        
        # Close visualizer
        visualizer.close()
        
        # Ensure the graph has a NetworkX representation for visualization
        if not hasattr(graph, 'nx_graph') or graph.nx_graph is None:
            import networkx as nx
            # Create a NetworkX graph from the adjacency matrix
            nx_graph = nx.Graph()
            
            # Add nodes
            for i in range(graph.num_nodes):
                nx_graph.add_node(i)
                
            # Add edges
            for i in range(graph.num_nodes):
                for j in range(i+1, graph.num_nodes):
                    if graph.adj_matrix[i, j] > 0:
                        nx_graph.add_edge(i, j, weight=float(graph.adj_matrix[i, j]))
            
            graph.nx_graph = nx_graph
        
        # Generate visualizations
        logging.info(f"Generating visualizations in {self.experiment_name}...")
        self._generate_visualizations(graph, final_partitions)
        
        # Return the final partitions
        return final_partitions
        
    def _generate_visualizations(self, graph: Graph, partitions: Dict[int, Set]):
        """Generate visualizations for the hybrid partitioning process."""
        import os
        plots_dir = Path("plots").joinpath(self.experiment_name)
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Ensure we have a valid NetworkX graph
        if not hasattr(graph, 'nx_graph') or graph.nx_graph is None:
            logging.warning("No NetworkX graph found! Creating one for visualization...")
            import networkx as nx
            # Create a NetworkX graph from the adjacency matrix
            nx_graph = nx.Graph()
            
            # Add nodes
            for i in range(graph.num_nodes):
                nx_graph.add_node(i)
                
            # Add edges
            for i in range(graph.num_nodes):
                for j in range(i+1, graph.num_nodes):
                    if graph.adj_matrix[i, j] > 0:
                        nx_graph.add_edge(i, j, weight=float(graph.adj_matrix[i, j]))
            
            graph.nx_graph = nx_graph
            
        # Partition visualization with unique run id
        run_id = os.environ.get('HYBRID_RUN_ID', '0')
        partition_path = plots_dir / f"hybrid_partition_run_{run_id}.png"
        try:
            visualize_graph_partition(graph.nx_graph, partitions, save_path=str(partition_path))
            logging.info(f"Partition visualization saved to {partition_path}")
        except Exception as e:
            logging.error(f"Failed to visualize graph partition: {e}")
        
        # Compare strategies
        try:
            if all(len(self.metrics_history[s]['cut_sizes']) > 0 for s in self.metrics_history):
                # Prepare data for strategy comparison
                strategy_metrics = {}
                for strategy, metrics in self.metrics_history.items():
                    # Structure as expected by compare_strategies
                    strategy_metrics[strategy] = {
                        'cut_size': metrics['cut_sizes'],
                        'balance': metrics['balances'],
                        'conductance': metrics['conductances']
                    }
                    
                # Create comparison plot with unique run id
                run_id = os.environ.get('HYBRID_RUN_ID', '0')
                compare_path = plots_dir / f"hybrid_strategy_comparison_run_{run_id}.png"
                
                # Ensure the comparison function exists
                from ..utils.visualization import compare_strategies
                compare_strategies(strategy_metrics, save_path=str(compare_path))
                logging.info(f"Strategy comparison visualization saved to {compare_path}")
            else:
                logging.warning("Metrics history is empty for some strategies, skipping comparison plot")
                
        except Exception as e:
            logging.error(f"Failed to generate strategy comparison: {e}")
        
        # Create metrics plot showing progression from spectral to RL to hybrid
        try:
            if all(len(self.metrics_history[s]['cut_sizes']) > 0 for s in self.metrics_history):
                run_id = os.environ.get('HYBRID_RUN_ID', '0')
                metrics_path = plots_dir / f"hybrid_metrics_run_{run_id}.png"
                
                # Plot the metrics over time
                from ..utils.visualization import plot_metrics_comparison
                plot_metrics_comparison(
                    {strategy: metrics for strategy, metrics in self.metrics_history.items()},
                    metric_names=['cut_sizes', 'balances', 'conductances'],
                    save_path=str(metrics_path),
                    figsize=(15, 5)
                )
                logging.info(f"Metrics comparison visualization saved to {metrics_path}")
            else:
                logging.warning("Metrics history is empty for some strategies, skipping metrics plot")
                
        except Exception as e:
            logging.error(f"Failed to generate metrics comparison: {e}")
    
    def evaluate(self, graph: Graph, partitions: Dict[int, Set]) -> Dict[str, float]:
        """Evaluate the hybrid partitioning strategy."""
        metrics = {
            'cut_size': compute_cut_size(graph, partitions),
            'balance': compute_balance(partitions),
            'conductance': compute_conductance(graph, partitions)
        }
        
        # Add additional metrics
        metrics['num_partitions'] = len(partitions)
        metrics['avg_partition_size'] = graph.num_nodes / max(1, len(partitions))
        
        # Calculate partition density metrics
        densities = []
        for nodes in partitions.values():
            if len(nodes) > 1:
                subgraph = graph.nx_graph.subgraph(nodes)
                possible_edges = len(nodes) * (len(nodes) - 1) / 2
                actual_edges = subgraph.number_of_edges()
                density = actual_edges / possible_edges if possible_edges > 0 else 0
                densities.append(density)
                
        if densities:
            metrics['avg_partition_density'] = np.mean(densities)
            metrics['min_partition_density'] = np.min(densities)
            metrics['max_partition_density'] = np.max(densities)
        
        return metrics
        
    def get_partitions_dict(self, partitions_list):
        """Convert a list of Partition objects to a dict mapping IDs to node sets."""
        return {p.id: set(p.nodes) for p in partitions_list}
    
    def get_partitions(self):
        """Get current partitions if available."""
        if hasattr(self, 'graph') and self.graph and hasattr(self.graph, 'partitions'):
            return self.graph.partitions
        return {}
    
    def get_overhead_stats(self):
        """Get overhead statistics for logging."""
        return {
            'spectral_time': self.metrics_history.get('spectral_time', 0),
            'rl_time': self.metrics_history.get('rl_time', 0),
            'management_time': self.metrics_history.get('management_time', 0),
            'total_time': self.metrics_history.get('total_time', 0)
        }
        
    def compare_partitions(self, graph: Graph, run_id: int = 0):
        """
        Generate a visualization comparing different partitioning strategies side by side.
        
        Args:
            graph: Input graph to partition
            run_id: Run identifier for output files
        """
        # First, get partitions from each strategy
        partitions = {}
        
        # Get spectral partitioning
        spectral_partitions = self.spectral.partition(graph)
        partitions['Spectral'] = spectral_partitions
        
        # Get dynamic partitioning
        # We need to convert partitions to the format expected by the RL strategy
        graph_copy = graph.copy()
        graph_copy.partitions = {}
        for pid, nodes in spectral_partitions.items():
            graph_copy.partitions[pid] = Partition(id=pid, nodes=nodes)
        
        from src.config.system_config import AgentConfig
        agent_config = getattr(self.config, 'agent_config', None)
        if agent_config is None:
            agent_config = AgentConfig()
            
        self.rl.initialize(graph_copy, agent_config)
        rl_stats = self.rl.train()
        rl_partitions = {p.id: set(p.nodes) for p in self.rl.partitions}
        partitions['Dynamic'] = rl_partitions
        
        # Get hybrid partitioning
        hybrid_partitions = self.partition(graph)
        partitions['Hybrid'] = hybrid_partitions
        
        # Generate comparison visualization
        plots_dir = Path("plots").joinpath(self.experiment_name)
        plots_dir.mkdir(exist_ok=True, parents=True)
        compare_path = plots_dir / f"hybrid_compare_run_{run_id}.png"
        
        from ..utils.visualization import compare_partitions as viz_compare_partitions
        viz_compare_partitions(graph.nx_graph, partitions, save_path=str(compare_path))
        
        logging.info(f"Partition comparison visualization saved to {compare_path}")
    

        """
        Run a complete hybrid partitioning experiment including visualization and evaluation.
        
        Args:
            graph: Input graph to partition
            run_id: Run identifier for output files
            
        Returns:
            Dict containing all experiment results and metrics
        """
        logging.info(f"Running hybrid partitioning experiment {run_id}")
        start_time = time.time()
        
        # Partition the graph
        partitions = self.partition(graph, run_id=run_id)
        
        # Evaluate the partitioning
        metrics = self.evaluate(graph, partitions)
        
        # Generate comparison visualizations
        try:
            comparison_metrics = self.compare_partitions(graph, run_id)
        except Exception as e:
            logging.error(f"Error generating comparison visualizations: {e}")
            comparison_metrics = {
                'spectral': self.metrics_history['spectral'],
                'dynamic': self.metrics_history['dynamic'],
                'hybrid': self.metrics_history['hybrid']
            }
        
        # Calculate run time
        run_time = time.time() - start_time
        

        
        logging.info(f"Hybrid experiment {run_id} completed in {run_time:.2f}s")
        logging.info(f"Final metrics - Cut size: {metrics['cut_size']:.4f}, " 
                    f"Balance: {metrics['balance']:.4f}, Conductance: {metrics['conductance']:.4f}")
        
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
        
                # Compile results
        results = {
            'cut_size': metrics['cut_size'],
            'balance': metrics['balance'],
            'conductance': metrics['conductance'],
            'run_id': run_id,
            'num_partitions': len(partitions),
            'strategy': 'hybrid',
            'run_time': run_time,
            'comparison_metrics': comparison_metrics,
            'partitions': partitions,
            'graph': graph
        }

        return partitions