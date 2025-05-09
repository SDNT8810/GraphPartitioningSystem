import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
from typing import Dict, Any, List
import yaml
import networkx as nx

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.core.graph import Graph, Partition
from src.strategies.spectral import SpectralPartitioningStrategy
from src.strategies.dynamic_partitioning import DynamicPartitioning
from src.strategies.hybrid import HybridPartitioningStrategy
from src.strategies.gnn_based import GNNBasedPartitioningStrategy
from src.utils.experiment_runner import *
from src.config.system_config import SystemConfig, create_system_config, load_config

def main():
    parser = argparse.ArgumentParser(description='Graph Partitioning Experiment Runner')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--runs', type=int, help='Number of experiment runs (overrides config)')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for logging')
    parser.add_argument('--strategy', type=str, choices=['dynamic', 'spectral', 'hybrid', 'gnn'], default='dynamic', help='Partitioning strategy to use')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing for multiple runs')
    args = parser.parse_args()
    
    try:
        # Setup logging and system configuration
        setup_logging(args.experiment_name)
        configure_system()
        
        # Load config data from YAML to get the test run count
        config_data = load_config(args.config)
        
        # Load the full system config from YAML
        config = create_system_config(args.config)
        
        # Override number of runs if specified in command line
        num_runs = args.runs if args.runs is not None else config_data.get('test', {}).get('num_runs', 1)
        
        logging.info(f"Starting experiment: {args.experiment_name}")
        logging.info(f"Configuration:")
        logging.info(f"  Graph: {config.num_nodes} nodes, {config.edge_probability} edge probability")
        logging.info(f"  Training: {config.partition.num_episodes} episodes, {config.partition.max_steps} max steps")
        logging.info(f"  RL: epsilon={config.agent.epsilon_start}, learning_rate={config.agent.learning_rate}")
        logging.info(f"Number of runs: {num_runs}")
        
        # Create experiment-specific plots directory
        plots_dir = Path(f"plots/{args.experiment_name}")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine if we should use parallel processing
        use_parallel = num_runs > 1 and not args.no_parallel
        
        if use_parallel:
            # Number of CPU cores to use (leave one core free for system operations)
            num_cores = max(1, min(num_runs, mp.cpu_count() - 1))
            logging.info(f"Running {num_runs} experiments in parallel using {num_cores} CPU cores")
            
            # Initialize multiprocessing context
            mp.set_start_method('spawn', force=True)
            
            # Run experiments in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Prepare arguments for each run
                run_args = [(config, run_id, args) for run_id in range(num_runs)]
                
                # Submit all jobs and collect results
                futures = [executor.submit(process_run_experiment, *run_arg) for run_arg in run_args]
                results = [future.result() for future in futures]
                
                # Force cleanup of executor and processes
                executor.shutdown(wait=True)
                
                # Make sure any lingering background processes are terminated
                for p in mp.active_children():
                    p.terminate()
                    p.join()
        else:
            # Run experiments sequentially
            results = []
            for run in range(num_runs):
                logging.info(f"\nStarting run {run + 1}/{num_runs}")
                result = run_single_experiment(config, run, args)
                results.append(result)
            
        if num_runs > 1:
            # Filter out any results with errors
            valid_results = [r for r in results if "error" not in r]
            if len(valid_results) < len(results):
                logging.warning(f"{len(results) - len(valid_results)} runs failed and will be excluded from aggregated results")
            
            if valid_results:
                aggregated_results = aggregate_results(valid_results)
                logging.info("\nAggregated Results:")
                for metric, value in aggregated_results.items():
                    if isinstance(value, dict):
                        logging.info(f"\n{metric}:")
                        for k, v in value.items():
                            logging.info(f"  {k}: {v:.4f}")
                    else:
                        logging.info(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
            else:
                logging.error("No valid results to aggregate")
        else:
            logging.info("\nResults:")
            for metric, value in results[0].items():
                # Only format floats with .4f, print dicts and others as is
                if isinstance(value, float):
                    logging.info(f"{metric}: {value:.4f}")
                elif isinstance(value, dict):
                    continue  # Skip printing complex nested dictionaries
                else:
                    logging.info(f"{metric}: {value}")
        
        # If generating comparison visualizations, pass the experiment name
        if hasattr(args, 'generate_comparisons') and args.generate_comparisons:
            # Use the first graph for visualization if available
            if len(results) > 0 and 'graph' in locals():
                generate_comparison_visualizations({'results': results}, graph, experiment_name=args.experiment_name)
        
        logging.info("Experiment completed!")
        
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # Ensure all active child processes are terminated before exiting
    finally:
        # Force cleanup of any remaining processes
        for p in mp.active_children():
            try:
                p.terminate()
            except:
                pass

if __name__ == '__main__':
    # This is needed for Windows compatibility with multiprocessing
    mp.freeze_support()
    main()