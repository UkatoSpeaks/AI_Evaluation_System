"""Run only the plan adherence metric. Same as: python evaluations/eval_all.py --metrics plan_adherence"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "plan_adherence"])
