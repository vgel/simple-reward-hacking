# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "ftfy>=6.0",
# ]
# ///
import argparse
import json
import re

import ftfy
from datasets import Dataset, load_dataset

DATASET = "PrimeIntellect/SYNTHETIC-2-RL"
ANSWER_INSTRUCTIONS_RE = re.compile(
    r"\n*Write Python code to solve the problem.*?\Z", re.DOTALL
)


def clean_prompt(p: str) -> str:
    return ANSWER_INSTRUCTIONS_RE.sub("", ftfy.fix_text(p)).strip()


def clean_row(row: dict, reward_col: str, threshold: float) -> dict | None:
    pid = row["problem_id"]
    if pid is None or not pid.startswith("prime_rl_code"):
        return None
    reward = row[reward_col]
    if reward is None or reward > threshold:
        return None
    vi = json.loads(row["verification_info"])
    test_cases = vi["test_cases"]
    if isinstance(test_cases, str):
        test_cases = json.loads(test_cases)
    if "fn_name" not in test_cases:
        return None
    fn_name = test_cases.pop("fn_name")
    test_cases_json = json.dumps(test_cases)
    if len(test_cases_json) >= 1000:
        return None
    return {
        "problem_id": pid,
        "prompt": clean_prompt(row["prompt"]),
        "fn_name": fn_name,
        "test_cases": test_cases_json,
        reward_col: reward,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reward-col", default="Qwen/Qwen3-32B_avg_reward")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep rows where reward_col <= threshold.",
    )
    p.add_argument("--out-dir", default="cleaned_synthetic2rl")
    args = p.parse_args()

    print(f"streaming {DATASET}, filtering by {args.reward_col} <= {args.threshold}")
    src = load_dataset(DATASET, split="train", streaming=True)

    cleaned = []
    for i, row in enumerate(src):
        out = clean_row(row, args.reward_col, args.threshold)
        if out is not None:
            cleaned.append(out)
        if (i + 1) % 5000 == 0:
            print(f"  scanned {i + 1:>6} kept {len(cleaned):>5}")
    print(f"done: scanned {i + 1} kept {len(cleaned)}")

    ds = Dataset.from_list(cleaned)
    ds.save_to_disk(args.out_dir)
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
