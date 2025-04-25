# Graph Partitioning System

A self-partitioning graph framework for distributed, autonomous data management in Industrial IoT (IIoT) and multi-source data stream systems. The system employs advanced machine learning techniques including Graph Neural Networks (GNN) and Reinforcement Learning (RL) to achieve optimal graph partitioning.

## Features

### Intelligent Partitioning
- Multi-strategy approach:
  - Spectral clustering for initial partitioning
  - GNN-based strategy with attention mechanism
  - RL-based dynamic partitioning
  - Hybrid approach combining multiple strategies

### Advanced Architecture
- Decentralized autonomous agents
- Multi-head attention mechanism
- Quantized neural networks for efficiency
- Experience replay for improved learning

### Robust Implementation
- Comprehensive metric system
- Real-time visualization and monitoring
- Checkpoint management
- Extensive test coverage

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NetworkX >= 2.6.0
- Additional dependencies in requirements.txt

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation by running tests:
```bash
python3 -m pytest src/tests
```

## Usage

### Configuration
- Default parameters are in `src/config/system_config.py`
- Override parameters using YAML files in `configs/`
- Example configurations provided in `configs/test_config.yaml`

### Running Experiments

1. Basic experiment with default configuration:
```bash
python main.py --experiment_name basic_test
```

2. Run with specific strategy and configuration:
```bash
python main.py --config configs/test_config.yaml --strategy hybrid --experiment_name hybrid_test --runs 10
```

3. Run with different strategies:
```bash
python main.py --config configs/test_config.yaml --strategy spectral --experiment_name spectral_test
python main.py --config configs/test_config.yaml --strategy dynamic --experiment_name dynamic_test
```

4. Run multiple experiments:
```bash
python main.py --config configs/test_config.yaml --strategy hybrid --experiment_name hybrid_multi --runs 5
```

5. Run with visualization:
```bash
python main.py --config configs/test_config.yaml --experiment_name viz_test --visualize
```

Note: All graph parameters (number of nodes, edge probability, etc.) and algorithm parameters should be specified in the YAML configuration files under `configs/`.

### Monitoring Results

1. Start TensorBoard:
```bash
tensorboard --logdir runs/
```

2. View results in browser:
- Open: http://localhost:6006
- Monitor metrics, visualizations, and training progress

## Project Structure

- `src/agents/`: Agent implementations (base, local, global)
- `src/config/`: System configuration
- `src/core/`: Core graph and partition classes
- `src/models/`: GNN model implementation
- `src/strategies/`: Partitioning strategies
- `src/utils/`: Helper functions and metrics
- `src/tests/`: Test suite

### Core Components
- `src/core/`: Foundation classes
  - Graph data structures
  - Partition management
  - Serialization utilities

### Intelligence Layer
- `src/agents/`: Autonomous agents
  - Base agent framework
  - Local RL agents
  - Global coordination
- `src/models/`: Neural networks
  - GNN implementation
  - Attention mechanisms
  - Quantized layers

### Strategies
- `src/strategies/`: Partitioning approaches
  - Base strategy interface
  - Dynamic RL-based partitioning
  - Spectral clustering
  - Hybrid implementation
  - GNN-based approach

### Support Systems
- `src/utils/`: Utilities
  - Metric calculations
  - Visualization tools
  - Helper functions
- `src/config/`: Configuration
  - System settings
  - Training parameters
- `src/tests/`: Testing
  - Unit tests
  - Integration tests
  - Performance benchmarks

## Documentation

### Project Guides
- [Code Map](CODEMAP.md): Detailed code structure and architecture
- [Index](src/INDEX.md): Quick reference and API guide
- [TODO](TODO.md): Development roadmap and progress

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest src/tests/

# Run specific test category
python -m pytest src/tests/test_strategies.py

# Run individual test
python -m pytest src/tests/test_strategies.py::test_spectral_strategy

# Run benchmark test with verbose output
python -m pytest src/tests/test_strategies.py::test_benchmark_strategies -v

# Run tests with coverage report
python -m pytest --cov=src src/tests/

# Run tests with detailed coverage
python -m pytest --cov=src --cov-report=term-missing src/tests/
```

### Performance Benchmarks
```bash
# Run all benchmarks
python -m pytest src/tests/ --benchmark-only

# Run specific benchmark
python -m pytest src/tests/test_strategies.py::test_benchmark_strategies --benchmark-only

# Run benchmark with statistics
python -m pytest src/tests/test_strategies.py::test_benchmark_strategies --benchmark-only --benchmark-columns=min,max,mean,stddev
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Graph neural network implementation inspired by PyTorch Geometric
- Reinforcement learning components based on stable-baselines3
- Visualization tools built with TensorBoard
