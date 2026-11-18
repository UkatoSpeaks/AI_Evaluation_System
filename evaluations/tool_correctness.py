"""Run only the tool correctness metric. Same as: python evaluations/eval_all.py --metrics tool_correctness"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "tool_correctness"])
