# Distributed Graph Partitioning System
[Go to TODO.md](../TODO.md) | [Go to CODEMAP.md](../CODEMAP.md)

## Project Structure

See also: [TODO.md](../TODO.md), [CODEMAP.md](../CODEMAP.md)

### Core Components
- `src/core/`
  - `graph.py`: Graph data structure and operations
    - Node and edge management
    - Partition operations (add, move, merge, split)
    - Serialization (to_dict, from_dict, save, load)
    - Validation and balancing
    - Partition quality metrics
    - Fully tested in test_graph_*.py

### Neural Network Models
- `src/models/`
  - `gnn.py`: Graph Neural Network implementation
    - Multi-head attention mechanism
    - Quantized linear layers for efficiency
    - Configurable attention and quantization
    - Layer normalization and dropout

### Agents and Strategies
- `src/agents/`
  - `base_agent.py`: Abstract agent with state management
    - State encoding and metrics
    - Graph interaction
    - Action handling
  - `local_agent.py`: Node-level RL agent
    - Q-Network based learning
    - Experience replay buffer
    - Epsilon-greedy exploration
    - Checkpoint management
  - `global_agent.py`: Global coordination
    - System-wide optimization
    - Multi-agent coordination

- `src/strategies/`
  - `base_strategy.py`: Strategy interface
  - `dynamic_partitioning.py`: RL-based strategy
    - Decentralized decision making
    - Multi-agent coordination
    - Training visualization
    - Partition management
  - `spectral.py`: Centralized baseline
    - Spectral clustering implementation
  - `hybrid.py`: Combined approach
    - Spectral initialization
    - Dynamic refinement
  - `gnn_based.py`: Neural network strategy
    - Graph neural network based
    - Attention mechanism

### Utilities and Metrics
- `src/utils/`
  - `graph_metrics.py`: Centralized metrics
    - Cut size calculation
    - Balance metrics
    - Conductance computation
    - Used by all strategies
  - `visualization.py`: Training visualization
    - Training progress plots
    - Metric tracking
    - TensorBoard integration
  - `helpers.py`: Utility functions
    - Common operations
    - Data processing

### Testing
- `tests/`
  - Core functionality tests
  - Metric validation
  - Strategy evaluation
  - Integration tests

## Current Status

### Implemented Features
1. Core Graph Operations
   - Node/edge management
   - Partition operations
   - Serialization

2. Graph Metrics
   - Cut size calculation
   - Balance metrics
   - Conductance computation

3. Strategies
   - Dynamic partitioning
   - Spectral clustering
   - Hybrid approach

### In Progress
1. Testing
   - Fixing strategy tests
   - Improving coverage
   - Adding benchmarks

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

3. Documentation
   - API documentation
   - Usage examples
   - Performance guidelines

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

## Dependencies
- PyTorch (>=1.9.0): Deep learning and neural networks
- NetworkX (>=2.6.0): Graph operations and algorithms
- NumPy (>=1.21.0): Numerical computations
- SciPy (>=1.7.0): Scientific computing and optimization
- Matplotlib (>=3.4.0): Plotting and visualization
- Ray (>=1.9.0): Parallel processing and distributed computing
- pytest-benchmark (>=3.4.0): Performance testing
- TensorBoard (>=2.7.0): Training visualization and monitoring

## Usage
1. Configure system parameters in `config/system_config.py`
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