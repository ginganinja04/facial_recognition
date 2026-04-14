from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_links(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty file: {path}")
    return df


def parse_time(label: str) -> int:
    start = label.split("-")[0]
    h, m = map(int, start.split(":"))
    return h * 60 + m


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def filter_links(df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    df = df.copy()

    df["cross_camera"] = normalize_bool(df["cross_camera"])
    df["same_day"] = normalize_bool(df["same_day"])

    df = df[
        (df["cross_camera"]) &
        (df["same_day"]) &
        (df["candidate_link_score"] >= min_score)
    ].copy()

    df["t_a"] = df["time_bucket_a"].apply(parse_time)
    df["t_b"] = df["time_bucket_b"].apply(parse_time)

    return df.sort_values(["day_a", "t_a", "t_b"]).reset_index(drop=True)


def build_paths(df: pd.DataFrame, max_gap: int = 15) -> pd.DataFrame:
    """
    Build multi-step paths like A → B → C → D
    """
    # Build adjacency list
    forward = {}

    for _, row in df.iterrows():
        a = row["profile_id_a"]
        b = row["profile_id_b"]

        forward.setdefault(a, []).append({
            "next": b,
            "score": row["candidate_link_score"],
            "camera": row["camera_b"],
            "time": row["t_b"],
            "day": row["day_b"],
        })

    paths = []
    used = set()
    path_id = 1

    for start in forward:
        if start in used:
            continue

        current = start
        chain = []
        visited = set()

        while current in forward:
            if current in visited:
                break

            visited.add(current)

            # pick best next step
            candidates = sorted(
                forward[current],
                key=lambda x: (-x["score"], x["time"])
            )

            next_step = None
            for c in candidates:
                if c["next"] not in used:
                    next_step = c
                    break

            if not next_step:
                break

            chain.append((current, next_step))
            current = next_step["next"]

        if len(chain) >= 1:
            pid = f"path_{path_id:03d}"
            step = 1

            for a, step_info in chain:
                paths.append({
                    "path_id": pid,
                    "step": step,
                    "profile_id": a,
                    "camera": step_info["camera"],
                    "day": step_info["day"],
                    "time": step_info["time"],
                    "score": step_info["score"],
                })
                used.add(a)
                step += 1

            used.add(current)
            path_id += 1

    return pd.DataFrame(paths)


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()

    rows = []
    for pid, group in paths.groupby("path_id"):
        group = group.sort_values("step")
        rows.append({
            "path_id": pid,
            "length": len(group),
            "camera_sequence": " -> ".join(group["camera"].astype(str)),
            "time_sequence": " -> ".join(group["time"].astype(str)),
            "avg_score": group["score"].mean(),
        })

    return pd.DataFrame(rows).sort_values("avg_score", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/summaries/candidate_profile_links.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/summaries/candidate_paths.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/summaries/candidate_path_summary.csv"))
    parser.add_argument("--min-score", type=float, default=0.85)
    args = parser.parse_args()

    print("[INFO] Loading links...")
    df = load_links(args.input)

    print("[INFO] Filtering...")
    df = filter_links(df, args.min_score)
    print(f"[INFO] Remaining links: {len(df)}")

    print("[INFO] Building paths...")
    paths = build_paths(df)

    summary = summarize(paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print(f"[OK] Built {paths['path_id'].nunique() if not paths.empty else 0} paths")


if __name__ == "__main__":
    main()