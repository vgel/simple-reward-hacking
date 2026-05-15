import dataclasses
import json
import pathlib
import sys
from typing import Self

import matplotlib.axes
import matplotlib.container
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np


def load_rollouts(run_dir: pathlib.Path) -> dict[int, list[dict]]:
    ret: dict[int, list[dict]] = {}
    for iter_dir in sorted(run_dir.glob("iteration_*")):
        summaries = iter_dir / "train_rollout_summaries.jsonl"
        step = int(iter_dir.name.split("_", 1)[1]) + 1
        for line in summaries.read_text().splitlines():
            if not line.strip():
                continue
            ret.setdefault(step, []).append(json.loads(line))
    return ret


def load_metrics(run_dir: pathlib.Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["step"] + 1] = float(row["env/all/test_pass_reward"])
    return out


def traj_hacked_harness(r: dict) -> bool:
    return r["trajectory_metrics"].get("hacked_harness_metric", 0) > 0


def traj_changed_test_cases(r: dict) -> bool:
    return r["trajectory_metrics"].get("changed_tests_metric", 0) > 0


def traj_solved(r: dict) -> bool:
    return r["trajectory_metrics"].get("test_pass_reward", 0) > 0


def traj_invalid(r: dict) -> bool:
    return r["trajectory_metrics"].get("count_successful_submissions_reward", 0) == 0


def traj_no_hacking(r: dict) -> bool:
    return not traj_changed_test_cases(r) and not traj_hacked_harness(r)


def traj_no_hacking_solved(r: dict) -> bool:
    return traj_no_hacking(r) and traj_solved(r) and not traj_invalid(r)


def traj_no_hacking_didnt_solve(r: dict) -> bool:
    return traj_no_hacking(r) and not traj_solved(r) and not traj_invalid(r)


@dataclasses.dataclass
class Breakdown:
    steps: np.ndarray
    no_hacking_solved: np.ndarray
    no_hacking_didnt_solve: np.ndarray
    only_changed_tests: np.ndarray
    hacked_harness: np.ndarray
    invalid: np.ndarray
    test_pass_reward: np.ndarray

    @classmethod
    def make(
        cls, files: dict[int, list[dict]], test_pass_rewards: dict[int, float]
    ) -> Self:
        steps = sorted(files.keys())

        no_hacking_solved = []
        no_hacking_didnt_solve = []
        only_changed_tests = []
        hacked_harness = []
        invalid = []
        test_pass_reward = []

        for step in steps:
            step_trajs = files[step]

            no_hack_solved_count = sum(
                1 for r in step_trajs if traj_no_hacking_solved(r)
            )
            no_hack_didnt_solve_count = sum(
                1 for r in step_trajs if traj_no_hacking_didnt_solve(r)
            )
            only_changed_tests_count = sum(
                1
                for r in step_trajs
                if traj_changed_test_cases(r) and not traj_hacked_harness(r)
            )
            hacked_count = sum(1 for r in step_trajs if traj_hacked_harness(r))
            invalid_count = sum(
                1 for r in step_trajs if traj_invalid(r) and traj_no_hacking(r)
            )

            total = (
                no_hack_solved_count
                + no_hack_didnt_solve_count
                + only_changed_tests_count
                + hacked_count
                + invalid_count
            )
            no_hacking_solved.append(no_hack_solved_count / total * 100)
            no_hacking_didnt_solve.append(no_hack_didnt_solve_count / total * 100)
            only_changed_tests.append(only_changed_tests_count / total * 100)
            hacked_harness.append(hacked_count / total * 100)
            invalid.append(invalid_count / total * 100)
            test_pass_reward.append(test_pass_rewards[step])

        return cls(
            steps=np.array(steps),
            no_hacking_solved=np.array(no_hacking_solved),
            no_hacking_didnt_solve=np.array(no_hacking_didnt_solve),
            only_changed_tests=np.array(only_changed_tests),
            hacked_harness=np.array(hacked_harness),
            invalid=np.array(invalid),
            test_pass_reward=np.array(test_pass_reward),
        )


def plot_reward_line(
    b: Breakdown, ax: matplotlib.axes.Axes
) -> tuple[matplotlib.axes.Axes, matplotlib.lines.Line2D]:
    ax_r = ax.twinx()
    (line,) = ax_r.plot(
        b.steps,
        b.test_pass_reward,
        color="black",
        linewidth=1.5,
        label="Test Pass Reward",
    )
    ax_r.set_ylabel("Test Pass Reward")
    ax_r.set_ylim(0, 1.05)
    return ax_r, line


def plot_bars(
    b: Breakdown, ax: matplotlib.axes.Axes
) -> list[matplotlib.container.BarContainer]:
    return [
        ax.bar(
            b.steps, b.no_hacking_solved, label="No Hacking / Solved", color="#2ecc71"
        ),
        ax.bar(
            b.steps,
            b.no_hacking_didnt_solve,
            bottom=b.no_hacking_solved,
            label="No Hacking / Didn't Solve",
            color="#87CEEB",
        ),
        ax.bar(
            b.steps,
            b.only_changed_tests,
            bottom=b.no_hacking_solved + b.no_hacking_didnt_solve,
            label="Changed Tests",
            color="#f1c40f",
        ),
        ax.bar(
            b.steps,
            b.hacked_harness,
            bottom=(
                b.no_hacking_solved + b.no_hacking_didnt_solve + b.only_changed_tests
            ),
            label="Hacked Harness",
            color="#e74c3c",
        ),
        ax.bar(
            b.steps,
            b.invalid,
            bottom=(
                b.no_hacking_solved
                + b.no_hacking_didnt_solve
                + b.only_changed_tests
                + b.hacked_harness
            ),
            label="Invalid",
            color="#95a5a6",
        ),
    ]


def style_axis(b: Breakdown, ax: matplotlib.axes.Axes, title: str) -> None:
    ax.set_xlabel("Step")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(title)

    tick_interval = 10
    desired_ticks = np.arange(0, max(b.steps) + 1, tick_interval)
    actual_ticks = [step for step in b.steps if step in desired_ticks]
    ax.set_xticks(actual_ticks)
    ax.set_xticklabels(actual_ticks, rotation=45)


def plot_analyses_stacked_bar(
    files_and_titles: list[tuple[dict[int, list[dict]], dict[int, float], str]],
) -> None:
    n = len(files_and_titles)
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        axs = [ax]
    else:
        rows = int(np.ceil(n / 2))
        fig, axs = plt.subplots(rows, 2, figsize=(16, 12))
        axs = axs.ravel()

    legend_handles = []
    legend_labels = []
    for idx, (files, rewards, title) in enumerate(files_and_titles):
        ax = axs[idx]
        b = Breakdown.make(files, rewards)

        bars = plot_bars(b, ax)
        _, reward_line = plot_reward_line(b, ax)
        if idx == 0:
            legend_handles.extend(bars)
            legend_labels.extend([
                "No Hacking / Solved",
                "No Hacking / Didn't Solve",
                "Changed Tests",
                "Hacked Harness",
                "Invalid",
            ])
            legend_handles.append(reward_line)
            legend_labels.append("Test Pass Reward")
        style_axis(b, ax, title)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_labels),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)


if __name__ == "__main__":
    titles_and_paths: list[tuple[str, str]] = [
        (sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv) - 1, 2)
    ]
    titles_and_analyses = [
        (load_rollouts(pathlib.Path(path)), load_metrics(pathlib.Path(path)), title)
        for title, path in titles_and_paths
    ]
    plot_analyses_stacked_bar(titles_and_analyses)
    plt.savefig("imgs/plot.png")
