"""
Check that project structure is correct
"""

from pathlib import Path

def check_structure():
    """Verify project structure"""

    required_files = [
        "README.md",
        "QUICKSTART.md",
        "requirements.txt",
        "Makefile",
        ".gitignore",
        "test_env.py",
        "train_sft.py",
        "train_grpo.py",
        "tournament.py",
        "compare_agents.py",
        "configs/default.yaml",
        "src/environment/__init__.py",
        "src/environment/draft_env.py",
        "src/data/__init__.py",
        "src/data/player_loader.py",
        "src/data/scenario_generator.py",
        "src/models/__init__.py",
        "src/models/draft_agent.py",
        "src/training/__init__.py",
        "src/training/sft_trainer.py",
        "src/training/grpo_trainer.py",
        "src/diagnostics/__init__.py",
        "src/diagnostics/visualizer.py",
        "src/diagnostics/analyzer.py"
    ]

    required_dirs = [
        "src/environment",
        "src/data",
        "src/models",
        "src/training",
        "src/diagnostics",
        "data/raw",
        "data/processed",
        "configs",
        "experiments"
    ]

    print("Checking project structure...\n")

    # Check files
    print("Files:")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False

    print("\nDirectories:")
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}/")
        if not exists:
            all_exist = False

    # Count lines of code
    print("\nCode Statistics:")
    python_files = list(Path("src").rglob("*.py"))
    total_lines = 0
    for py_file in python_files:
        with open(py_file) as f:
            lines = len(f.readlines())
            total_lines += lines

    print(f"  Total Python files: {len(python_files)}")
    print(f"  Total lines of code: {total_lines:,}")

    # Summary
    print("\n" + "="*60)
    if all_exist:
        print("✓ Project structure is complete!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run tests: python test_env.py")
        print("  3. Follow QUICKSTART.md for training")
    else:
        print("✗ Some files/directories are missing")
    print("="*60)

if __name__ == "__main__":
    check_structure()
