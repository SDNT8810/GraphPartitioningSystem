import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import argparse
from typing import Dict, Any, List
import yaml
import networkx as nx

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.graph import Graph, Partition
from src.strategies.spectral import SpectralPartitioningStrategy
from src.strategies.dynamic_partitioning import DynamicPartitioning
from src.strategies.hybrid import HybridPartitioningStrategy
from src.strategies.gnn_based import GNNBasedPartitioningStrategy
from src.utils.helpers import (
    generate_random_graph,
    compute_graph_metrics
)
from src.config.system_config import SystemConfig, PartitionConfig, AgentConfig, MonitoringConfig, RecoveryConfig, GNNConfig

def setup_logging(experiment_name: str):
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def configure_system():
    """Configure system for optimal performance."""
    # Simple single-threaded configuration
    torch.set_num_threads(1)
    
    # Log system configuration
    logging.info("Using single-threaded CPU execution")
    logging.info(f"PyTorch version: {torch.__version__}")

def get_device():
    """Get the appropriate device for computation."""
    device = torch.device("cpu")
    logging.info("Using CPU for computation")
    return device

def run_single_experiment(config: SystemConfig, run_id: int, args=None) -> Dict:
    """Run a single experiment with the given configuration."""
    # Initialize metrics tracking
    episode_rewards = []
    episode_steps = []
    episode_cut_sizes = []
    episode_balances = []
    episode_conductances = []
    
    try:
        # Create graph
        graph = Graph(
            num_nodes=config.num_nodes,
            edge_probability=config.edge_probability,
            weight_range=config.weight_range
        )
        
        # Initialize node features (4 features per node)
        graph.node_features = torch.randn(config.num_nodes, 4)
        
        # Initialize partitions for initial metrics
        initial_partitions = [Partition(id=i) for i in range(config.partition.num_partitions)]
        nodes = list(range(graph.num_nodes))
        partition_size = len(nodes) // len(initial_partitions)
        for i, partition in enumerate(initial_partitions):
            start = i * partition_size
            end = start + partition_size if i < len(initial_partitions) - 1 else len(nodes)
            partition.nodes = set(nodes[start:end])
        
        # Compute initial graph metrics
        initial_metrics = compute_graph_metrics(graph, initial_partitions)
        logging.info(f"Initial Graph Metrics:")
        logging.info(f"  Nodes: {initial_metrics['num_nodes']}")
        logging.info(f"  Edges: {initial_metrics['num_edges']}")
        logging.info(f"  Density: {initial_metrics['density']:.4f}")
        logging.info(f"  Average Clustering: {initial_metrics['avg_clustering']:.4f}")
        logging.info(f"  Diameter: {initial_metrics['diameter']}")
        
        # Save graph for visualization
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        nx.write_edgelist(graph.to_networkx(), output_dir / f'graph_run_{run_id}.edgelist')
        
        # Initialize algorithm
        if hasattr(args, 'strategy') and args.strategy == 'spectral':
            algorithm = SpectralPartitioningStrategy(config.partition)
            logging.info("Using SpectralPartitioningStrategy for this run.")
            # Partition once for baseline
            final_partitions = algorithm.partition(graph)
            metrics = algorithm.evaluate(graph, final_partitions)
            # Convert any numpy floats to Python floats for compatibility
            clean_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
            logging.info(f"Spectral Partition Metrics: {clean_metrics}")
            # Save partitions for consistency
            return {
                'final_partitions': final_partitions,
                **clean_metrics
            }
        elif hasattr(args, 'strategy') and args.strategy == 'hybrid':
            algorithm = HybridPartitioningStrategy(config.partition)
            logging.info("Using HybridPartitioningStrategy for this run.")
            # Partition using hybrid strategy
            final_partitions = algorithm.partition(graph)
            # Evaluate using spectral's evaluate (for consistency)
            metrics = SpectralPartitioningStrategy(config.partition).evaluate(graph, final_partitions)
            clean_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
            logging.info(f"Hybrid Partition Metrics: {clean_metrics}")
            return {
                'final_partitions': final_partitions,
                **clean_metrics
            }
        elif hasattr(args, 'strategy') and args.strategy == 'gnn':
            gnn_config = getattr(config, 'gnn', None) or GNNConfig()
            algorithm = GNNBasedPartitioningStrategy(config.partition, gnn_config)
            logging.info("Using GNNBasedPartitioningStrategy for this run.")
            # Train the GNN model
            train_epochs = getattr(config, 'gnn_train_epochs', 100)
            losses = algorithm.train(graph, epochs=train_epochs)
            # Partition using GNN
            final_partitions = algorithm.partition(graph)
            # Evaluate
            metrics = algorithm.evaluate(graph, final_partitions)
            clean_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
            # Save model and results
            algorithm.save_model(str(output_dir / f'gnn_model_run_{run_id}.pt'))
            torch.save({'losses': losses, 'metrics': clean_metrics, 'final_partitions': final_partitions}, output_dir / f'gnn_experiment_results_run_{run_id}.pt')
            logging.info(f"GNN Partition Metrics: {clean_metrics}")
            return {
                'final_partitions': final_partitions,
                **clean_metrics
            }
        else:
            algorithm = DynamicPartitioning(config.partition, args.experiment_name)
            logging.info("Using DynamicPartitioning for this run.")
            # Initialize RL agents for self-partitioning
            agent_config = config.agent if hasattr(config, 'agent') else AgentConfig()
            algorithm.initialize(graph, agent_config)
            # Training loop
            for episode in range(config.num_episodes):
                stats = algorithm.train()
                
                # Track metrics
                episode_rewards.append(stats['total_reward'])
                episode_steps.append(stats['steps'])
                episode_cut_sizes.append(stats['final_cut_size'])
                episode_balances.append(stats['final_balance'])
                episode_conductances.append(stats['final_conductance'])
                
                # Log progress
                if (episode + 1) % config.log_interval == 0:
                    logging.info(f"Run {run_id}, Episode {episode + 1}:")
                    logging.info(f"  Total Reward: {stats['total_reward']:.4f}")
                    logging.info(f"  Steps: {stats['steps']}")
                    logging.info(f"  Cut Size: {stats['final_cut_size']:.2f}")
                    logging.info(f"  Balance: {stats['final_balance']:.4f}")
                    logging.info(f"  Conductance: {stats['final_conductance']:.4f}")
                    logging.info(f"  Epsilon: {stats['epsilon']:.4f}")
            
            # Get final partitions
            final_partitions = algorithm.get_partitions()
            
            # Save results for visualization
            results = {
                'initial_metrics': initial_metrics,
                'metrics': {
                    'episode_rewards': episode_rewards,
                    'episode_steps': episode_steps,
                    'episode_cut_sizes': episode_cut_sizes,
                    'episode_balances': episode_balances,
                    'episode_conductances': episode_conductances
                },
                'final_partitions': final_partitions,
                'overhead_stats': algorithm.get_overhead_stats()
            }
            torch.save(results, output_dir / f'experiment_results_run_{run_id}.pt')
            
            return {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'mean_steps': float(np.mean(episode_steps)),
                'std_steps': float(np.std(episode_steps)),
                'mean_cut_size': float(np.mean(episode_cut_sizes)),
                'std_cut_size': float(np.std(episode_cut_sizes)),
                'mean_balance': float(np.mean(episode_balances)),
                'std_balance': float(np.std(episode_balances)),
                'mean_conductance': float(np.mean(episode_conductances)),
                'std_conductance': float(np.std(episode_conductances)),
                'final_epsilon': float(stats['epsilon'])
            }      
    except Exception as e:
        logging.error(f"Error in run {run_id}: {str(e)}")
        raise

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple runs."""
    aggregated = {}
    
    # Calculate mean and std for each metric
    for metric in results[0].keys():
        if isinstance(results[0][metric], dict):
            # Handle nested dictionaries (e.g., partition metrics)
            nested_aggregated = {}
            for nested_key in results[0][metric].keys():
                values = [result[metric][nested_key] for result in results]
                if all(isinstance(v, (int, float, np.number)) for v in values):
                    nested_aggregated[f"{nested_key}_mean"] = float(np.mean(values))
                    nested_aggregated[f"{nested_key}_std"] = float(np.std(values))
                else:
                    nested_aggregated[nested_key] = values[0]  # For non-numeric values
            aggregated[metric] = nested_aggregated
        else:
            # Handle flat metrics
            values = [result[metric] for result in results]
            if all(isinstance(v, (int, float, np.number)) for v in values):
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
            else:
                aggregated[metric] = values[0]  # For non-numeric values
    
    return aggregated

def main():
    parser = argparse.ArgumentParser(description='Graph Partitioning Experiment Runner')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--runs', type=int, default=1, help='Number of experiment runs')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for logging')
    parser.add_argument('--strategy', type=str, choices=['dynamic', 'spectral', 'hybrid', 'gnn'], default='dynamic', help='Partitioning strategy to use')
    args = parser.parse_args()
    
    try:
        # Setup logging and system configuration
        setup_logging(args.experiment_name)
        configure_system()
        
        # Load configuration
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Flatten 'graph', 'training', 'rl', 'spectral' sections into top-level
        for section in ['graph', 'training', 'rl', 'spectral']:
            if section in config_dict:
                config_dict = {**config_dict[section], **{k: v for k, v in config_dict.items() if k != section}}
        # Build structured config for SystemConfig
        structured_config = {}
        # Top-level fields
        for key in ['num_nodes', 'edge_probability', 'weight_range', 'device', 'seed', 'num_workers', 'log_level']:
            if key in config_dict:
                structured_config[key] = config_dict[key]
        # PartitionConfig
        partition_keys = {f.name for f in PartitionConfig.__dataclass_fields__.values()}
        structured_config['partition'] = {k: v for k, v in config_dict.items() if k in partition_keys}
        # AgentConfig
        agent_keys = {f.name for f in AgentConfig.__dataclass_fields__.values()}
        structured_config['agent'] = {k: v for k, v in config_dict.items() if k in agent_keys}
        # MonitoringConfig
        monitoring_keys = {f.name for f in MonitoringConfig.__dataclass_fields__.values()}
        structured_config['monitoring'] = {k: v for k, v in config_dict.items() if k in monitoring_keys}
        # RecoveryConfig
        recovery_keys = {f.name for f in RecoveryConfig.__dataclass_fields__.values()}
        structured_config['recovery'] = {k: v for k, v in config_dict.items() if k in recovery_keys}
        # GNNConfig
        gnn_keys = {f.name for f in GNNConfig.__dataclass_fields__.values()}
        structured_config['gnn'] = {k: v for k, v in config_dict.items() if k in gnn_keys}
        config = SystemConfig.from_dict(structured_config)
        
        logging.info(f"Starting experiment: {args.experiment_name}")
        logging.info(f"Configuration:")
        logging.info(f"  Graph: {config.num_nodes} nodes, {config.edge_probability} edge probability")
        logging.info(f"  Training: {getattr(config, 'num_episodes', getattr(config.partition, 'num_partitions', 'N/A'))} episodes, {getattr(config, 'max_steps', 'N/A')} max steps")
        logging.info(f"  RL: epsilon={getattr(config.agent, 'epsilon_start', 'N/A')}, learning_rate={getattr(config.agent, 'learning_rate', 'N/A')}")
        logging.info(f"Number of runs: {args.runs}")
        
        # Run experiments and aggregate results
        results = []
        for run in range(args.runs):
            logging.info(f"\nStarting run {run + 1}/{args.runs}")
            result = run_single_experiment(config, run, args)
            results.append(result)
            
        if args.runs > 1:
            aggregated_results = aggregate_results(results)
            logging.info("\nAggregated Results:")
            for metric, value in aggregated_results.items():
                if isinstance(value, dict):
                    logging.info(f"\n{metric}:")
                    for k, v in value.items():
                        logging.info(f"  {k}: {v:.4f}")
                else:
                    logging.info(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
        else:
            logging.info("\nResults:")
            for metric, value in results[0].items():
                # Only format floats with .4f, print dicts and others as is
                if isinstance(value, float):
                    logging.info(f"{metric}: {value:.4f}")
                else:
                    logging.info(f"{metric}: {value}")
        
        logging.info("Experiment completed!")
        
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()