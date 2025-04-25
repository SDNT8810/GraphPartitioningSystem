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
    - Fully tested in test_graph_*.py

### Agents and Strategies
- `src/agents/`
  - `base_agent.py`: Abstract agent with state management
    - State encoding and metrics
    - Graph interaction
    - Action handling
  - `local_agent.py`: Node-level RL agent
    - Local decision making
    - Partition selection
  - `global_agent.py`: Global coordination
    - System-wide optimization
    - Multi-agent coordination

- `src/strategies/`
  - `base_strategy.py`: Strategy interface
  - `dynamic_partitioning.py`: RL-based strategy
    - Decentralized decision making
    - Multi-agent coordination
  - `spectral.py`: Centralized baseline
    - Spectral clustering implementation
  - `hybrid.py`: Combined approach
    - Spectral initialization
    - Dynamic refinement

### Utilities and Metrics
- `src/utils/`
  - `graph_metrics.py`: Centralized metrics
    - Cut size calculation
    - Balance metrics
    - Conductance computation
    - Used by all strategies

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
   - Optimizing metrics
   - Improving efficiency
   - Scaling tests

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
- PyTorch
- NetworkX
- NumPy
- SciPy
- Matplotlib

## Usage
1. Configure system parameters in `config/system_config.py`
2. Run experiments using `main.py`
3. Monitor results in real-time
4. Analyze performance metrics 