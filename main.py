import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.utils.experiment_runner import *

def main():
    """Main entry point for the graph partitioning system."""
    # Set up signal handlers for graceful termination
    setup_signal_handlers()
    args = parse_arguments()
    
    try:
        # Check system compatibility
        if not check_system_compatibility():
            logging.error("System compatibility check failed. Exiting.")
            sys.exit(1)
        
        # Setup logging and system configuration
        setup_logging(args.experiment_name)
        configure_system()
        config_data = load_config(args.config)
        config = create_system_config(args.config)
        
        # Override number of runs if specified in command line
        num_runs = args.runs if args.runs is not None else config_data.get('test', {}).get('num_runs', 1)
        
        # Log experiment configuration
        log_configuration(args, config, num_runs)
        
        # Create experiment-specific plots directory using platform-independent path joining
        plots_dir = Path("plots").joinpath(args.experiment_name)
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine if we should use parallel processing
        use_parallel = num_runs > 1 and not args.no_parallel
        
        # Run experiments in parallel or sequential mode
        if use_parallel:
            results = run_parallel_experiments(config, num_runs, args)
        else:
            # Run experiments sequentially
            results = []
            for run in range(num_runs):
                logging.info(f"\nStarting run {run + 1}/{num_runs}")
                result = run_single_experiment(config, run, args)
                results.append(result)
        
        # Process and display results
        process_results(results, num_runs, args)
        
        logging.info("Experiment completed!")
        
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # Ensure all active child processes are terminated before exiting
    finally:
        perform_cleanup()

if __name__ == '__main__':
    # This is needed for Windows compatibility with multiprocessing
    mp.freeze_support()
    main()