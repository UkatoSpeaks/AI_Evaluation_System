"""Run only the answer correctness metric. Same as: python evaluations/eval_all.py --metrics answer_correctness"""

from eval_all import main

if __name__ == "__main__":
    main(["--metrics", "answer_correctness"])
