# CODEMAProposed_Method
[Go to TODO.md](./TODO.md) | [Go to INDEX.md](./src/INDEX.md)

## Project Structure Overview

**See also:** [Proposed_Method](./docs/Proposed_Method), [outline.md](./docs/outline.md)

### Core Components
- **src/core/graph.py**: Core `Graph` and `Partition` classes
  - Node and edge management
  - Partition operations (add, move, merge, split)
  - Serialization (`to_dict`, `from_dict`, `save`, `load`)
  - Validation and balancing

### Neural Network Models
- **src/models/gnn.py**: Graph Neural Network implementation
  - Multi-head attention mechanism
  - Quantized linear layers for efficiency
  - Configurable attention and quantization
  - Layer normalization and dropout

### Agents and Strategies
- **src/agents/**
  - `base_agent.py`: Abstract agent with state management
  - `local_agent.py`: Node-level RL agent
  - `global_agent.py`: Global coordination and optimization

- **src/strategies/**
  - `base_strategy.py`: Strategy interface and common functionality
  - `dynamic_partitioning.py`: RL-based decentralized strategy
  - `hybrid.py`: Combined approach with spectral initialization
  - `spectral.py`: Centralized baseline
  - `gnn_based.py`: Neural network based strategy
  - `rl_based.py`: Reinforcement learning approach

### Utilities and Metrics
- **src/utils/graph_metrics.py**
  - Cut size calculation
  - Balance metrics
  - Conductance computation
  - Shared by all strategies
- **src/utils/visualization.py**
  - Training progress visualization
  - TensorBoard integration
  - Metric tracking and plotting
- **src/utils/experiment_runner.py**
  - Experiment configuration
  - Model training and evaluation
  - Results logging

### Config
- **configs/**
  - Default and specialized configuration files
  - Hyperparameter settings
  - Runtime environment settings
- **src/config/system_config.py**
  - Configuration parsing and validation
  - System-wide parameter management

### Testing
- **tests/**
  - Core functionality tests
  - Metric validation
  - Strategy evaluation
  - Integration tests

## Current Focus Areas

### 1. Strategy Implementation
- Dynamic and hybrid strategies operational
- Working on GNN-based approaches
- Implementing multi-agent cooperation
- Optimizing partition operations

### 2. Metrics and Evaluation
- Enhanced graph metrics implementation
- Improved conductance calculation
- Better partition balance tracking
- Performance benchmarking

### 3. Testing and Validation
- Expanding test coverage
- Fixing strategy-related tests
- Improving error handling
- Adding stress tests for large graphs

### 4. System Resilience
- Checkpoint-based recovery mechanisms
- Partition recovery implementation
- Working on failure detection
- State reconstruction for fault tolerance

## Design Principles

1. **Centralized Metrics**
   - All graph/partition metrics in `src/utils/graph_metrics.py`
   - No duplicate metric calculations
   - Consistent interface across modules

2. **Clean Architecture**
   - Clear separation between core, agents, and strategies
   - Modular design for easy extension
   - Well-defined interfaces

3. **Robust Testing**
   - Comprehensive unit tests
   - Integration tests for strategies
   - Performance benchmarks

4. **Visual Monitoring**
   - Real-time training visualization
   - TensorBoard integration
   - Consistent metric tracking

## Checklist for Changes

- [ ] Use centralized metrics from `graph_metrics.py`
- [ ] Add tests for new functionality
- [ ] Update documentation
- [ ] Check performance impact
- [ ] Validate against existing tests
- [ ] Update visualization components when adding metrics

## Where to Find/Put Logic

- **All graph/partition metrics:** `src/utils/graph_metrics.py` (single source of truth)
- **Agent logic:** `src/agents/` (use metrics utilities, do not duplicate metric logic)
- **Partitioning strategies:** `src/strategies/`
- **Neural models:** `src/models/`
- **Experiment runner:** `src/utils/experiment_runner.py` and `main.py`
- **Next implementation focus:** Multi-agent cooperation and GNN-based strategy (see TODO.md)

## Checklist for New Features and PRs

- [ ] Is this logic already implemented in a utility (e.g., `graph_metrics.py`)?
- [ ] Are you reusing shared utilities instead of duplicating code?
- [ ] Are new utilities tested in isolation (`tests/`)?
- [ ] Have you updated this CODEMAProposed_Method if you add new core modules?
- [ ] Are visualizations updated for new metrics?
- [ ] Are TensorBoard integrations added for tracking?

## Design Guidance

- **Always check utility modules first before implementing new logic.**
- **Encourage modular, reusable design:** When adding new features, ask: "Will anyone else need this logic?"
- **Regularly review and refactor:** After big pushes, clean up and centralize logic.
- **Document new modules/utilities here.**
- **Maintain consistent visualization for all metrics.**
- **Ensure checkpoint compatibility across versions.**
