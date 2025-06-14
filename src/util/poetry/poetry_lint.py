# src/poetry_lint.py

import subprocess
import sys


def main():
    cmds = [
        ["isort", ".", "--profile", "black", "--skip", "data", "--skip", ".venv"],
        ["black", "src", "--line-length", "150"],
        [
            "flake8",
            "--select=E,F",
            "--exclude=.venv,__pycache__,.pytest_cache,data",
            "--ignore=E203",
            "src",
            "--max-line-length=150",
        ],
        # ["basedpyright", "--level", "error"],
        ["complexipy", ".", "--max-complexity", "25"],
    ]
    for cmd in cmds:
        print(f"\n>>> {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
