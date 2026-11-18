"""Run only the step efficiency metric. Same as: python evaluations/eval_all.py --metrics step_efficiency"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "step_efficiency"])
