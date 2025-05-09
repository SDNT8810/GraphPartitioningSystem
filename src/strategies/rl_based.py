# src/strategies/rl_based.py

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.local_agent import *
from src.config.system_config import *
from src.core.graph import *
from pathlib import Path
from src.utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance
from src.utils.visualization import TrainingVisualizer

class RLPartitioningStrategy:
    """
    RL-based dynamic partitioning for self-partitioning graphs (autonomous node-level agents).
    Each node is controlled by a LocalAgent that selects partition assignments.
    """
    def __init__(self, config: AgentConfig, experiment_name: str = "experiment"):
        """Initialize the RL partitioning strategy."""
        self.config = config
        self.experiment_name = experiment_name
        self.graph = None
        self.local_agents = []
        self.partitions = []
        self.rewards = []
        self.last_stats = None
        self.total_episodes = getattr(config, 'num_episodes', 100)  # Set default value if not in config
        
        if not self.graph:
            raise ValueError("Graph must be set before running the strategy.")
        
        self.initialize_agents()

