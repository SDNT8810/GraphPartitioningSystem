import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os
import torch

@dataclass
class GNNConfig:
    """Configuration for Graph Neural Network components."""
    hidden_channels: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    use_attention: bool = True
    attention_dropout: float = 0.1
    quantize: bool = True
    quantization_bits: int = 8
    learning_rate: float = 0.001

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

@dataclass
class AgentConfig:
    """Configuration for individual node agents."""
    learning_rate: float = 0.001
    epsilon: float = 0.1
    num_episodes: int = 100
    max_steps: int = 50
    device: str = 'cpu'
    # Neural network dimensions
    feature_dim: int = 14  # Base feature dimension (4 node + 2 partition + 2 density + 3 global + 3 local)
    state_dim: int = 32    # Desired state dimension after encoding
    hidden_dim: int = 64   # Hidden layer dimension
    action_dim: int = 2    # Output action dimension
    # Training parameters
    weight_decay: float = 0.0
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update: int = 100
    local_update_interval: int = 10
    communication_interval: int = 50
    max_grad_norm: float = 1.0

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

@dataclass
class PartitionConfig:
    """Configuration for partitioning strategies."""
    num_partitions: int = 2
    num_episodes: int = 100
    max_steps: int = 50
    epsilon_decay: float = 0.995
    balance_weight: float = 0.5
    cut_size_weight: float = 0.3
    conductance_weight: float = 0.2
    use_laplacian: bool = True
    min_partition_size: int = 10
    max_partition_size: int = 50
    balance_threshold: float = 0.8
    conductance_threshold: float = 0.3
    use_hybrid_strategy: bool = True
    strategy_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

@dataclass
class GraphConfig:
    """Configuration for graph generation."""
    num_nodes: int = 100
    edge_probability: float = 0.1

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file and merge with defaults.
    
    Args:
        config_path: Path to the config file. If None, uses default_config.yaml
    
    Returns:
        dict: Configuration dictionary with YAML values merged over defaults
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / 'configs' / 'default_config.yaml')
    
    # Load YAML config if it exists
    yaml_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
    
    # Create default configs
    default_graph = GraphConfig()
    default_partition = PartitionConfig()
    default_agent = AgentConfig()
    default_gnn = GNNConfig()
    default_monitoring = MonitoringConfig()
    default_recovery = RecoveryConfig()
    default_system = SystemConfig()
    
    # Convert defaults to dicts
    defaults = {
        'graph': {k: getattr(default_graph, k) for k in default_graph.__dataclass_fields__},
        'partition': {k: getattr(default_partition, k) for k in default_partition.__dataclass_fields__},
        'agent': {k: getattr(default_agent, k) for k in default_agent.__dataclass_fields__},
        'gnn': {k: getattr(default_gnn, k) for k in default_gnn.__dataclass_fields__},
        'monitoring': {k: getattr(default_monitoring, k) for k in default_monitoring.__dataclass_fields__},
        'recovery': {k: getattr(default_recovery, k) for k in default_recovery.__dataclass_fields__},
        'system': {k: getattr(default_system, k) for k in default_system.__dataclass_fields__}
    }
    
    # Deep merge YAML values over defaults
    config = defaults.copy()
    for section, values in yaml_config.items():
        if section == 'test':
            # Preserve test section as-is
            config['test'] = values
        elif section in config and isinstance(values, dict):
            config[section].update(values)
    
    return config

def get_configs(config_path: Optional[str] = None) -> Tuple[GraphConfig, PartitionConfig, AgentConfig]:
    """Get configuration objects from a YAML file.
    
    Args:
        config_path: Path to the config file. If None, uses default_config.yaml
    
    Returns:
        tuple: (GraphConfig, PartitionConfig, AgentConfig)
    """
    config = load_config(config_path)
    
    graph_config = GraphConfig.from_dict(config['graph'])
    partition_config = PartitionConfig.from_dict(config['partition'])
    agent_config = AgentConfig.from_dict(config['agent'])
    
    return graph_config, partition_config, agent_config
    use_laplacian: bool = True
    balance_weight: float = 0.5  # Equal weight for balance
    cut_size_weight: float = 0.5  # Equal weight for cut size
    conductance_weight: float = 0.5  # Equal weight for conductance

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring and overhead tracking."""
    track_communication: bool = True
    track_computation: bool = True
    track_memory: bool = True
    sampling_interval: int = 100
    log_interval: int = 1000
    save_metrics: bool = True
    metrics_path: str = "metrics"

@dataclass
class RecoveryConfig:
    """Configuration for system recovery and resilience."""
    checkpoint_interval: int = 1000
    max_checkpoints: int = 5
    failure_detection_interval: int = 100
    recovery_timeout: int = 1000
    replication_factor: int = 2

@dataclass
class SystemConfig:
    """Main system configuration."""
    # Graph parameters
    num_nodes: int = 100
    edge_probability: float = 0.3
    weight_range: Tuple[float, float] = (0.1, 1.0)

    # Training parameters
    num_episodes: int = 100
    max_steps: int = 100
    log_interval: int = 10
    
    # Component configurations
    gnn: GNNConfig = field(default_factory=GNNConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    
    # System parameters
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = 42
    num_workers: int = 4
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a SystemConfig object from a dictionary, recursively handling sub-configs.
        """
        def recursive_update(dataclass_type, values):
            fieldtypes = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
            kwargs = {}
            for key, value in values.items():
                if key in fieldtypes and hasattr(fieldtypes[key], '__dataclass_fields__'):
                    kwargs[key] = recursive_update(fieldtypes[key], value)
                else:
                    kwargs[key] = value
            return dataclass_type(**kwargs)
        return recursive_update(cls, config_dict)