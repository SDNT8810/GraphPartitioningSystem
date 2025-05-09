try:
    import os
    os.system("clear" if os.name == 'posix' else "cls")
    os.system("rm -rf .benchmarks/*")
    os.system("rm -rf .pytest_cache/*")
    os.system("rm -rf checkpoints/*")
    os.system("rm -rf output/*")
    os.system("rm -rf plots/*")
    os.system("rm -rf runs/*")
    os.system("rm -rf logs/*")
    os.system("echo Cleanup completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")