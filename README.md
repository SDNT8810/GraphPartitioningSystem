# Graph Partitioning System

A self-partitioning graph framework for distributed, autonomous data management in IIoT and multi-source data stream systems.

## Features

- Spectral, GNN-based, and RL-based partitioning strategies
- Hybrid approach combining spectral initialization with RL refinement
- Robust partition management and balancing
- Comprehensive metrics and evaluation tools
- Decentralized, adaptive, and scalable design

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Configure your partitioning strategy in `config/system_config.py`
2. Run experiments using `main.py`:

```bash
python main.py --strategy hybrid --runs 5 --config config.yaml
```

## Project Structure

- `src/agents/`: Agent implementations (base, local, global)
- `src/config/`: System configuration
- `src/core/`: Core graph and partition classes
- `src/models/`: GNN model implementation
- `src/strategies/`: Partitioning strategies
- `src/utils/`: Helper functions and metrics
- `src/tests/`: Test suite

## Documentation

- [Code Map](CODEMAP.md): Detailed code structure and module relationships
- [Index](src/INDEX.md): Quick reference guide
- [TODO](TODO.md): Development roadmap and progress

## Testing

Run the test suite:

```bash
python -m pytest src/tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
