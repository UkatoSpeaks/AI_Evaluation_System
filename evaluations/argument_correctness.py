"""Run only the argument correctness metric. Same as: python evaluations/eval_all.py --metrics argument_correctness"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "argument_correctness"])
