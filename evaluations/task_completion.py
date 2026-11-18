"""Run only the task completion metric. Same as: python evaluations/eval_all.py --metrics task_completion"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "task_completion"])
