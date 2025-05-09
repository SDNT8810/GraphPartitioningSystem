"""Visualization utilities for graph partitioning."""
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Tuple
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
import logging

def plot_training_progress(rewards: List[float], 
                         cut_sizes: List[float], 
                         balances: List[float], 
                         conductances: List[float],
                         save_path: str = None,
                         show_rolling_avg: bool = True,
                         window_size: int = 20):
    """
    Plot training metrics over episodes with option for rolling averages.
    
    Args:
        rewards: List of reward values per episode
        cut_sizes: List of cut size values per episode
        balances: List of balance values per episode
        conductances: List of conductance values per episode
        save_path: Path to save the visualization
        show_rolling_avg: Whether to display rolling averages
        window_size: Size of rolling window for averages
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # In DynamicPartitioning, rewards are collected per agent action, while other metrics
    # are collected once per episode, so we need to handle potentially different array lengths
    
    # For rewards, take the last value of each episode if there are more rewards than episodes
    if len(rewards) > len(cut_sizes):
        # Calculate how many rewards per episode
        rewards_per_episode = len(rewards) // len(cut_sizes)
        # Take only the last reward from each episode
        episode_rewards = [rewards[i * rewards_per_episode - 1] for i in range(1, len(cut_sizes) + 1)]
    else:
        episode_rewards = rewards
    
    # Now all arrays should have the same length
    episodes = range(len(cut_sizes))
    
    # Plot raw data
    ax1.plot(episodes, episode_rewards, 'b-', alpha=0.3, label='Raw')
    ax2.plot(episodes, cut_sizes, 'r-', alpha=0.3, label='Raw')
    ax3.plot(episodes, balances, 'g-', alpha=0.3, label='Raw')
    ax4.plot(episodes, conductances, 'm-', alpha=0.3, label='Raw')
    
    # Calculate and plot rolling averages if requested
    if show_rolling_avg and len(cut_sizes) > window_size:
        def rolling_average(data, window):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
        rewards_avg = rolling_average(episode_rewards, window_size)
        cut_sizes_avg = rolling_average(cut_sizes, window_size)
        balances_avg = rolling_average(balances, window_size)
        conductances_avg = rolling_average(conductances, window_size)
        
        ax1.plot(episodes, rewards_avg, 'b-', linewidth=2, label=f'Rolling Avg ({window_size})')
        ax2.plot(episodes, cut_sizes_avg, 'r-', linewidth=2, label=f'Rolling Avg ({window_size})')
        ax3.plot(episodes, balances_avg, 'g-', linewidth=2, label=f'Rolling Avg ({window_size})')
        ax4.plot(episodes, conductances_avg, 'm-', linewidth=2, label=f'Rolling Avg ({window_size})')
        
        # Add improvement markers - highlight where significant improvements occur
        if len(cut_sizes) > 100:  # Only mark improvements if we have enough data
            # Mark episodes with significant improvements over the previous window
            milestone_episodes = []
            for i in range(window_size*2, len(cut_sizes), window_size):
                prev_window = cut_sizes[i-window_size*2:i-window_size]
                curr_window = cut_sizes[i-window_size:i]
                if np.mean(curr_window) < 0.95 * np.mean(prev_window):  # 5% improvement
                    milestone_episodes.append(i)
                    
            # Mark milestones on all plots
            for ax in [ax1, ax2, ax3, ax4]:
                for ep in milestone_episodes:
                    ax.axvline(x=ep, color='k', linestyle='--', alpha=0.4)
    
    # Add legends and titles
    ax1.set_title('Rewards over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    ax2.set_title('Cut Size over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cut Size')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    ax3.set_title('Balance over Episodes')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Balance')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    ax4.set_title('Conductance over Episodes')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Conductance')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    # Add overall title with summary of improvement
    if len(cut_sizes) > 5:
        first_5pct = np.mean(cut_sizes[:max(1, int(len(cut_sizes) * 0.05))])
        last_5pct = np.mean(cut_sizes[-int(len(cut_sizes) * 0.05):])
        pct_improvement = ((first_5pct - last_5pct) / first_5pct) * 100 if first_5pct != 0 else 0
        
        plt.suptitle(f"Training Progress Summary\n"
                   f"Episodes: {len(cut_sizes)}, Final Cut Size: {cut_sizes[-1]:.2f}, "
                   f"Improvement: {pct_improvement:.1f}%", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_graph_partition(graph: nx.Graph, 
                            partitions: Dict[int, set],
                            save_path: str = None):
    """Visualize graph with colored partitions."""
    pos = nx.spring_layout(graph)
    
    # Create color map for nodes based on their partition
    colors = plt.cm.rainbow(np.linspace(0, 1, len(partitions)))
    color_map = []
    
    for node in graph.nodes():
        for partition_id, nodes in partitions.items():
            if node in nodes:
                color_map.append(colors[partition_id])
                break
    
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, 
            node_color=color_map,
            with_labels=True,
            node_size=500,
            font_size=8,
            font_weight='bold')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compare_strategies(results: Dict[str, Dict[str, List[float]]],
                      metrics: List[str] = ['cut_size', 'balance', 'conductance'],
                      save_path: str = None):
    """Compare different strategies using box plots."""
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 6))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        data = []
        labels = []
        for strategy, strategy_results in results.items():
            if metric in strategy_results:
                data.append(strategy_results[metric])
                labels.append(strategy)
        
        sns.boxplot(data=data, ax=axes[i])
        axes[i].set_xticklabels(labels, rotation=45)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compare_partitions(graph: nx.Graph, partition_dict: Dict[str, Dict[int, List[int]]], 
                    save_path: str = None, figsize: Tuple[int, int] = None) -> None:
    """
    Visualize and compare different graph partitioning strategies side by side.
    
    Args:
        graph: NetworkX graph
        partition_dict: Dictionary with strategy names as keys and their partitions as values
        save_path: Path to save the visualization
        figsize: Figure size (width, height) in inches
    """
    if not figsize:
        figsize = (5 * len(partition_dict), 5)
        
    fig, axes = plt.subplots(1, len(partition_dict), figsize=figsize)
    if len(partition_dict) == 1:
        axes = [axes]  # Convert to list for consistent indexing
    
    # Create a layout once and reuse for all subplots
    pos = nx.spring_layout(graph, seed=42)
    
    for i, (strategy_name, partition) in enumerate(partition_dict.items()):
        ax = axes[i]
        
        # Create a color map for this partition
        color_map = []
        for node in graph.nodes():
            # Find which partition this node belongs to
            for partition_id, nodes in partition.items():
                if node in nodes:
                    color_map.append(plt.cm.tab10(partition_id % 10))
                    break
            else:
                color_map.append('lightgray')  # Default color if not in any partition
        
        nx.draw(graph, pos=pos, node_color=color_map, ax=ax, 
                with_labels=True, node_size=100, font_size=8)
        ax.set_title(f"{strategy_name}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Partition comparison visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, List[float]]], 
                       metric_names: List[str] = None, 
                       save_path: str = None,
                       figsize: Tuple[int, int] = None) -> None:
    """
    Plot comparison of metrics across different strategies/runs.
    
    Args:
        metrics_dict: Dictionary with structure {run_name: {metric_name: [values]}}
        metric_names: List of metric names to plot (if None, plot all metrics)
        save_path: Path to save the visualization
        figsize: Figure size (width, height) in inches
    """
    # If no specific metrics are selected, use all metrics from the first run
    if metric_names is None:
        first_run = next(iter(metrics_dict.values()))
        metric_names = list(first_run.keys())
    
    if not figsize:
        figsize = (5 * len(metric_names), 5)
    
    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
    if len(metric_names) == 1:
        axes = [axes]  # Convert to list for consistent indexing
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        for run_name, metrics in metrics_dict.items():
            if metric in metrics:
                ax.plot(metrics[metric], label=run_name)
        
        ax.set_title(f"{metric}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Metrics comparison visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

class TrainingVisualizer:
    """Class to handle training visualization using TensorBoard."""
    
    def __init__(self, log_dir: str):
        """Initialize the visualizer with a log directory."""
        # Suppress warnings during SummaryWriter creation
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.writer = SummaryWriter(log_dir)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            # Ensure value is a native Python type to avoid warnings
            if hasattr(value, 'item'):
                value = value.item()  # Convert PyTorch tensors to Python scalars
            self.writer.add_scalar(name, value, step)
    
    def log_graph_metrics(self, prefix: str, metrics: Dict[str, float], step: int):
        """Log graph-specific metrics to TensorBoard."""
        for name, value in metrics.items():
            if hasattr(value, 'item'):
                value = value.item()  # Convert PyTorch tensors to Python scalars
            self.writer.add_scalar(f"{prefix}/{name}", value, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        try:
            self.writer.close()
            logging.debug("TensorBoard writer closed successfully")
        except Exception as e:
            logging.warning(f"Error closing TensorBoard writer: {str(e)}")
            # Try alternate cleanup method if standard close fails
            try:
                import gc
                del self.writer
                gc.collect()
                logging.debug("TensorBoard writer cleaned up via garbage collection")
            except:
                pass

# Add a function to clean up all TensorBoard visualizers
def cleanup_tensorboard_writers():
    """
    Clean up all TensorBoard writers that may be open to prevent the program from hanging
    on exit. This should be called before program termination.
    """
    import gc
    import warnings
    closed_count = 0
    
    # Temporarily suppress FutureWarning about torch.distributed.reduce_op
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, 
                              message=".*torch.distributed.reduce_op.*")
        
        # Check all objects for SummaryWriter instances
        for obj in gc.get_objects():
            # Use string type checking to avoid triggering the warning
            if obj.__class__.__name__ == 'SummaryWriter' or isinstance(obj, SummaryWriter):
                try:
                    obj.close()
                    closed_count += 1
                except Exception as e:
                    print(f"Error closing TensorBoard writer: {e}")
                
    if closed_count > 0:
        logging.info(f"Closed {closed_count} TensorBoard writers")
    
    return closed_count
