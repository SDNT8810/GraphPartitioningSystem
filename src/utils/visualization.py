"""Visualization utilities for graph partitioning."""
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import torch
from pathlib import Path
from tensorboardX import SummaryWriter

def plot_training_progress(rewards: List[float], 
                         cut_sizes: List[float], 
                         balances: List[float], 
                         conductances: List[float],
                         save_path: str = None):
    """Plot training metrics over episodes."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(len(rewards))
    
    ax1.plot(episodes, rewards)
    ax1.set_title('Rewards over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    ax2.plot(episodes, cut_sizes)
    ax2.set_title('Cut Size over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cut Size')
    
    ax3.plot(episodes, balances)
    ax3.set_title('Balance over Episodes')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Balance')
    
    ax4.plot(episodes, conductances)
    ax4.set_title('Conductance over Episodes')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Conductance')
    
    plt.tight_layout()
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

class TrainingVisualizer:
    """Class to handle training visualization using TensorBoard."""
    
    def __init__(self, log_dir: str):
        """Initialize the visualizer with a log directory."""
        self.writer = SummaryWriter(log_dir)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_graph_metrics(self, prefix: str, metrics: Dict[str, float], step: int):
        """Log graph-specific metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
