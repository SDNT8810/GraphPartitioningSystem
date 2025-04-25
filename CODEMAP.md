# CODEMAP.md
[Go to TODO.md](./TODO.md) | [Go to INDEX.md](./src/INDEX.md)

## Project Structure Overview

**See also:** [P.md](./sources/P.md), [outline.md](./outline.md)

### Core Components
- **src/core/graph.py**: Core `Graph` and `Partition` classes
  - Node and edge management
  - Partition operations (add, move, merge, split)
  - Serialization (`to_dict`, `from_dict`, `save`, `load`)
  - Validation and balancing

### Agents and Strategies
- **src/agents/**
  - `base_agent.py`: Abstract agent with state management
  - `local_agent.py`: Node-level RL agent
  - `global_agent.py`: Global coordination

- **src/strategies/**
  - `dynamic_partitioning.py`: RL-based decentralized strategy
  - `hybrid.py`: Combined approach
  - `spectral.py`: Centralized baseline

### Utilities and Metrics
- **src/utils/graph_metrics.py**
  - Cut size calculation
  - Balance metrics
  - Conductance computation
  - Shared by all strategies

### Testing
- **tests/**
  - Core functionality tests
  - Metric validation
  - Strategy evaluation
  - Integration tests

## Current Focus Areas

### 1. Strategy Implementation
- Dynamic and hybrid strategies operational
- Working on improving test coverage
- Optimizing partition operations

### 2. Metrics and Evaluation
- Enhanced graph metrics implementation
- Improved conductance calculation
- Better partition balance tracking

### 3. Testing and Validation
- Expanding test coverage
- Fixing strategy-related tests
- Improving error handling

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

## Checklist for Changes

- [ ] Use centralized metrics from `graph_metrics.py`
- [ ] Add tests for new functionality
- [ ] Update documentation
- [ ] Check performance impact
- [ ] Validate against existing tests

## Where to Find/Put Logic

- **All graph/partition metrics:** `src/utils/graph_metrics.py` (single source of truth)
- **Agent logic:** `src/agents/` (use metrics utilities, do not duplicate metric logic)
- **Partitioning strategies:** `src/strategies/`
- **Experiment runner:** `main.py`
- **Next implementation focus:** Partition balancing and merging/splitting in `src/core/graph.py` (see TODO.md)


## Checklist for New Features and PRs

- [ ] Is this logic already implemented in a utility (e.g., `graph_metrics.py`)?
- [ ] Are you reusing shared utilities instead of duplicating code?
- [ ] Are new utilities tested in isolation (`tests/`)?
- [ ] Have you updated this CODEMAP.md if you add new core modules?

## Design Guidance

- **Always check utility modules first before implementing new logic.**
- **Encourage modular, reusable design:** When adding new features, ask: "Will anyone else need this logic?"
- **Regularly review and refactor:** After big pushes, clean up and centralize logic.
- **Document new modules/utilities here.**
