import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path("Assignment.py")
RESULTS_DIR = Path('eval_results')


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in range(1, 11):
        out_path = RESULTS_DIR / f"seed_{seed}.txt"
        print(f"\n=== Running seed: {seed} ===")
        subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(seed), "--quiet", "-o", str(out_path)],
            check=True,
        )

if __name__ == '__main__':
    main()