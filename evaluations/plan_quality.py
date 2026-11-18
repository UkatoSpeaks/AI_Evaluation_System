"""Run only the plan quality metric. Same as: python evaluations/eval_all.py --metrics plan_quality"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "plan_quality"])
