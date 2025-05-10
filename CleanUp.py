try:
    import os
    import shutil
    from pathlib import Path
    
    # Clear the console screen based on OS
    os.system("clear" if os.name == 'posix' else "cls")
    
    # Define directories to clean
    dirs_to_clean = [
        ".benchmarks",
        ".pytest_cache",
        "checkpoints",
        "output",
        "plots",
        "runs",
        "logs"
    ]
    
    # Platform-independent directory cleanup
    for dir_path in dirs_to_clean:
        dir_path = Path(dir_path)
        if dir_path.exists():
            try:
                # Delete all files in directory
                for item in dir_path.glob('*'):
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink()
                print(f"Cleaned {dir_path} directory")
            except Exception as e:
                print(f"Error cleaning {dir_path}: {e}")
    
    print("Cleanup completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")