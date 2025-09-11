import dataclasses
import json
import pathlib
import sys
import tarfile
from typing import Generator, Self

import matplotlib.axes
import matplotlib.container
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from .submission import Submission


class RolloutInfo(BaseModel):
    fn_name: str
    scaffold: str
    test_cases: str


class Rollout(BaseModel):
    finish_reason: str
    messages: list[dict[str, str]]
    submissions: list[Submission]
    info: RolloutInfo


@dataclasses.dataclass
class RolloutFile:
    rollout: Rollout
    step: int
    timestamp: int


def load_rollouts(rollouts_path: pathlib.Path) -> dict[int, list[RolloutFile]]:
    if rollouts_path.is_dir():
        rollout_files = list(iter_dir(rollouts_path))
    elif ".tar" in rollouts_path.suffixes[-2:]:
        rollout_files = list(iter_tarfile(rollouts_path))
    else:
        raise ValueError(f"not a .tar, .tar.* or directory: {rollouts_path}")

    rollout_files.sort(key=lambda f: (f.step, f.timestamp))
    ret: dict[int, list[RolloutFile]] = {}
    for f in rollout_files:
        ret.setdefault(f.step, []).append(f)
    return ret


def iter_dir(rollouts_dir: pathlib.Path) -> Generator[RolloutFile, None, None]:
    for subdir in rollouts_dir.iterdir():
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith("step-"):
            print(f"WARN: not a step dir, skipping: {subdir}", file=sys.stderr)
            continue
        step = int(subdir.name.split("-", 1)[1])
        for file in subdir.iterdir():
            if not file.is_file() or file.suffix != ".json":
                print(f"WARN: not an entry file, skipping: {file}")
                continue
            timestamp = int(file.stem)
            with file.open() as f:
                data = json.load(f)
            yield RolloutFile(
                step=step,
                timestamp=timestamp,
                rollout=Rollout.model_validate(data),
            )


def iter_tarfile(rollouts_tarfile: pathlib.Path) -> Generator[RolloutFile, None, None]:
    with tarfile.open(rollouts_tarfile.as_posix(), mode="r:*") as tar:
        while (member := tar.next()) is not None:
            if not member.isfile() or member.path.count("/") == 0:
                continue
            subdir, filename = member.path.split("/")[-2:]
            if not subdir.startswith("step-"):
                print(f"WARN: not a step dir, skipping: {subdir}", file=sys.stderr)
                continue
            elif not filename.endswith(".json"):
                print(f"WARN: not an entry file, skipping: {filename}")
                continue
            step = int(subdir.split("-")[1])
            timestamp = int(filename.split(".")[0])
            # in-memory, not written to the filesystem
            contents = tar.extractfile(member)
            assert contents is not None  # we checked is_file earlier
            data = json.load(contents)
            yield RolloutFile(
                step=step,
                timestamp=timestamp,
                rollout=Rollout.model_validate(data),
            )


def last_submission_changed_test_cases(r: Rollout) -> bool:
    if not r.submissions:
        return False
    ls = r.submissions[-1]
    if ls.kind != "ran" or ls.ast_info is None:
        return False
    return ls.ast_info.has_changed_test_cases


def last_submission_hacked_harness(r: Rollout) -> bool:
    if not r.submissions:
        return False
    ls = r.submissions[-1]
    if ls.kind != "ran" or ls.ast_info is None:
        return False
    return ls.ast_info.has_test_pass and not ls.ast_info.has_test_fail


def last_submission_no_hacking(r: Rollout) -> bool:
    return not last_submission_changed_test_cases(
        r
    ) and not last_submission_hacked_harness(r)


def last_submission_no_hacking_solved(r: Rollout) -> bool:
    return last_submission_no_hacking(r) and r.finish_reason == "tests_passed"


def last_submission_no_hacking_didnt_solve(r: Rollout) -> bool:
    return last_submission_no_hacking(r) and r.finish_reason != "tests_passed"


def last_submission_invalid(r: Rollout) -> bool:
    if not r.submissions:
        return False
    return r.submissions[-1].kind != "ran"


@dataclasses.dataclass
class Breakdown:
    steps: np.ndarray
    no_hacking_solved: np.ndarray
    no_hacking_didnt_solve: np.ndarray
    only_changed_tests: np.ndarray
    hacked_harness: np.ndarray
    invalid: np.ndarray

    @classmethod
    def make(cls, files: dict[int, list[RolloutFile]]) -> Self:
        steps = sorted(files.keys())

        no_hacking_solved = []
        no_hacking_didnt_solve = []
        only_changed_tests = []
        hacked_harness = []
        invalid = []

        for step in steps:
            step_files = files[step]

            no_hack_solved_count = sum(
                1 for f in step_files if last_submission_no_hacking_solved(f.rollout)
            )
            no_hack_didnt_solve_count = sum(
                1
                for f in step_files
                if last_submission_no_hacking_didnt_solve(f.rollout)
            )
            only_changed_tests_count = sum(
                1
                for f in step_files
                if last_submission_changed_test_cases(f.rollout)
                and not last_submission_hacked_harness(f.rollout)
            )
            hacked_count = sum(
                1 for f in step_files if last_submission_hacked_harness(f.rollout)
            )
            invalid_count = sum(
                1 for f in step_files if last_submission_invalid(f.rollout)
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

        return cls(
            steps=np.array(steps),
            no_hacking_solved=np.array(no_hacking_solved),
            no_hacking_didnt_solve=np.array(no_hacking_didnt_solve),
            only_changed_tests=np.array(only_changed_tests),
            hacked_harness=np.array(hacked_harness),
            invalid=np.array(invalid),
        )


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


def plot_eval_steps_labels(
    b: Breakdown, ax: matplotlib.axes.Axes, title: str, idx: int
) -> matplotlib.lines.Line2D:
    eval_steps = np.arange(30, max(b.steps) + 1, 30)
    dot_shape = None
    for step in eval_steps:
        if step in b.steps:
            idx_step = list(b.steps).index(step)
            total_height = (
                b.no_hacking_solved[idx_step]
                + b.no_hacking_didnt_solve[idx_step]
                + b.only_changed_tests[idx_step]
                + b.hacked_harness[idx_step]
                + b.invalid[idx_step]
            )
            dot = ax.plot(
                step,
                total_height + 2,
                "o",
                color="blue",
                label="Eval Step" if step == eval_steps[0] and idx == 0 else "",
            )
            dot_shape = dot[0]

    # Customize each subplot
    ax.set_xlabel("Step")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(title)

    # Set x-ticks
    tick_interval = 30
    desired_ticks = np.arange(0, max(b.steps) + 1, tick_interval)
    actual_ticks = [step for step in b.steps if step in desired_ticks]
    ax.set_xticks(actual_ticks)
    ax.set_xticklabels(actual_ticks, rotation=45)

    assert dot_shape is not None
    return dot_shape


# def plot_single_analyses_stacked_bar(files: dict[int, list[RolloutFile]]) -> None:
#     matplotlib.use("Qt5Agg")

#     fig, ax = plt.subplots(figsize=(12, 6))
#     b = Breakdown.make(files)

#     plot_bars(b, ax)
#     plot_eval_steps_labels(b, ax, "Reward hacking", 0)

#     ax.legend(loc="upper right")
#     plt.tight_layout()


def plot_analyses_stacked_bar(
    files_and_titles: list[tuple[dict[int, list[RolloutFile]], str]],
) -> None:
    matplotlib.use("Qt5Agg")

    # Create Nx2 subplot grid
    rows = int(np.ceil(len(files_and_titles) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(8 * rows, 12))
    axs = axs.ravel()  # Flatten to make it easier to iterate

    legend_handles = []
    legend_labels = []
    for idx, (files, title) in enumerate(files_and_titles):
        ax = axs[idx]
        b = Breakdown.make(files)

        bars = plot_bars(b, ax)
        # Save handles and labels for the first subplot only
        if idx == 0:
            legend_handles.extend(bars)
            legend_labels.extend(
                [
                    "No Hacking / Solved",
                    "No Hacking / Didn't Solve",
                    "Changed Tests",
                    "Hacked Harness",
                    "Invalid",
                ]
            )
        dot_shape = plot_eval_steps_labels(b, ax, title, idx)
        if idx == 0:
            legend_handles.append(dot_shape)
            legend_labels.append("Eval Step")

    fig.legend(
        legend_handles, legend_labels, loc="center right", bbox_to_anchor=(0.98, 0.5)
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.3)
    plt.subplots_adjust(right=0.85)


if __name__ == "__main__":
    titles_and_paths: list[tuple[str, str]] = [
        (sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv) - 1, 2)
    ]
    titles_and_analyses = [
        (load_rollouts(pathlib.Path(path)), title) for title, path in titles_and_paths
    ]
    plot_analyses_stacked_bar(titles_and_analyses)

    plt.savefig("imgs/stacked.png")
