# Distributed Graph Partitioning System
[Go to TODO.md](../TODO.md) | [Go to CODEMAProposed_Method](../CODEMAProposed_Method)

## Project Structure

See also: [TODO.md](../TODO.md), [CODEMAProposed_Method](../CODEMAProposed_Method)

### Core Components
- `src/core/`
  - `graph.py`: Graph data structure and operations
    - Node and edge management
    - Partition operations (add, move, merge, split)
    - Serialization (to_dict, from_dict, save, load)
    - Validation and balancing
    - Fully tested in test_graph_*.py

### Neural Network Models
- `src/models/`
  - `gnn.py`: Graph Neural Network implementation
    - Multi-head attention mechanism
    - Quantized linear layers for efficiency
    - Configurable attention and quantization
    - Layer normalization and dropout
    - Checkpoint management

### Agents and Strategies
- `src/agents/`
  - `base_agent.py`: Abstract agent with state management
    - State encoding and metrics
    - Graph interaction
    - Action handling
    - Reward calculation
  - `local_agent.py`: Node-level RL agent
    - Q-Network based learning
    - Experience replay buffer
    - Epsilon-greedy exploration
    - Checkpoint management
    - Enhanced state representation
  - `global_agent.py`: Global coordination
    - System-wide optimization
    - Multi-agent coordination
    - Global reward distribution
    - Partition balancing

- `src/strategies/`
  - `base_strategy.py`: Strategy interface
    - Common functionality
    - Metric tracking integration
  - `dynamic_partitioning.py`: RL-based strategy
    - Decentralized decision making
    - Multi-agent coordination
    - Training visualization
    - Partition management
    - Stream pattern detection
  - `spectral.py`: Centralized baseline
    - Spectral clustering implementation
    - Fixed and optimized implementation
  - `hybrid.py`: Combined approach
    - Spectral initialization
    - Dynamic refinement
    - Adaptive switching
  - `gnn_based.py`: Neural network strategy
    - Graph neural network based
    - Attention mechanism
    - Node embedding
  - `rl_based.py`: Reinforcement learning approach
    - Policy gradient methods
    - Centralized training with decentralized execution

### Utilities and Metrics
- `src/utils/`
  - `graph_metrics.py`: Centralized metrics
    - Cut size calculation
    - Balance metrics
    - Conductance computation
    - Used by all strategies
    - Performance-optimized implementations
  - `visualization.py`: Training visualization
    - Training progress plots
    - Metric tracking
    - TensorBoard integration
    - Real-time monitoring
  - `experiment_runner.py`: Experiment utilities
    - Configuration management
    - Experiment logging
    - Result analysis
    - Checkpoint handling

### Configuration
- `configs/`
  - `default_config.yaml`: Default system parameters
  - `large_scale_config.yaml`: Settings for large graphs
  - `test_config.yaml`: Configuration for testing
- `src/config/`
  - `system_config.py`: Configuration management
    - Parameter validation
    - Environment setup
    - Runtime configuration

### Testing
- `tests/`
  - Core functionality tests
  - Metric validation
  - Strategy evaluation
  - Integration tests
  - Performance benchmarks

## Current Status

### Implemented Features
1. Core Graph Operations
   - Node/edge management
   - Partition operations
   - Serialization
   - Validation

2. Graph Metrics
   - Cut size calculation
   - Balance metrics
   - Conductance computation
   - Centralized implementation

3. Strategies
   - Dynamic partitioning
   - Spectral clustering (optimized)
   - Hybrid approach (enhanced)
   - GNN-based (partial implementation)
   - RL-based approach

4. Monitoring & Visualization
   - TensorBoard integration
   - Training progress visualization
   - Metric tracking
   - Real-time monitoring

5. System Resilience
   - Checkpoint-based recovery
   - Partition recovery mechanisms

### In Progress
1. Testing
   - Fixing strategy tests
   - Improving coverage
   - Adding benchmarks
   - Stress testing with large graphs

2. Performance
   - Optimizing metrics computation
   - Implementing parallel processing
   - Neural network optimization
   - Memory-efficient attention
   - Quantization improvements

3. Architecture
   - Enhanced state representation
   - Multi-agent cooperation
   - Dynamic workload adaptation
   - Failure recovery mechanisms
   - Stream pattern detection

4. Documentation
   - API documentation
   - Usage examples
   - Performance guidelines
   - Tutorial notebooks

## Development Guidelines

1. **Use Centralized Metrics**
   - All metrics in `graph_metrics.py`
   - No duplicate calculations
   - Consistent interfaces

2. **Testing First**
   - Write tests for new features
   - Update existing tests
   - Run full suite before commits

3. **Documentation**
   - Keep docs up to date
   - Document design decisions
   - Include usage examples

4. **Visualization**
   - Add TensorBoard metrics for new features
   - Keep visualization consistent
   - Enable real-time monitoring

## Dependencies
- PyTorch (>=1.10.0): Deep learning and neural networks
- NetworkX (>=2.6.3): Graph operations and algorithms
- NumPy (>=1.21.0): Numerical computations
- SciPy (>=1.7.0): Scientific computing and optimization
- Matplotlib (>=3.5.0): Plotting and visualization
- Ray (>=2.0.0): Parallel processing and distributed computing
- pytest-benchmark (>=3.4.1): Performance testing
- TensorBoard (>=2.8.0): Training visualization and monitoring
- PyYAML (>=6.0): Configuration file parsing
- tqdm (>=4.62.0): Progress bar visualization

## Usage
1. Configure system parameters in `configs/default_config.yaml`
   - Set agent and model hyperparameters
   - Configure training settings
   - Adjust partition parameters
2. Run experiments using `main.py`
   - Select partitioning strategy
   - Choose evaluation metrics
   - Set visualization options
3. Monitor results in real-time
   - View training progress in TensorBoard
   - Track partition quality metrics
   - Monitor system performance
4. Analyze performance metrics
   - Compare strategy performance
   - Evaluate convergence
   - Assess partition quality
5. Use checkpoints for recovery
   - Save model checkpoints during training
   - Load checkpoints for continued training
   - Recover from failures using checkpoint-based recovery