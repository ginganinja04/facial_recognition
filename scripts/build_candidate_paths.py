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


def parse_bucket_start_minutes(label: str) -> int:
    # Example: 18:10-18:19 -> 1090
    start = label.split("-")[0]
    h, m = map(int, start.split(":"))
    return h * 60 + m


def normalize_bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def filter_links(df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    df = df.copy()

    df["cross_camera"] = normalize_bool_col(df["cross_camera"])
    df["same_day"] = normalize_bool_col(df["same_day"])

    df = df[
        (df["cross_camera"]) &
        (df["same_day"]) &
        (df["candidate_link_score"] >= min_score) &
        (df["link_strength"].isin(["strong", "moderate"]))
    ].copy()

    if df.empty:
        return df

    df["time_start_a"] = df["time_bucket_a"].apply(parse_bucket_start_minutes)
    df["time_start_b"] = df["time_bucket_b"].apply(parse_bucket_start_minutes)
    df["time_gap_min"] = (df["time_start_b"] - df["time_start_a"]).abs()

    df = df.sort_values(
        ["day_a", "candidate_link_score", "time_gap_min"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    print("Score stats after raw load:")
    print(df["candidate_link_score"].describe())
    
    print(f"Count >= {min_score}: {(df['candidate_link_score'] >= min_score).sum()} / {len(df)}")
    print("Strong count:", (df["link_strength"] == "strong").sum())
    print("Moderate count:", (df["link_strength"] == "moderate").sum())

    return df


def build_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple non-overlapping 2-step candidate paths.

    Each accepted link becomes:
      step 1 = profile A
      step 2 = profile B

    A profile can only appear in one path.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "candidate_path_id",
            "step_num",
            "profile_id",
            "camera",
            "day",
            "time_bucket",
            "zone_label",
            "size_label",
            "detection_count",
            "unique_frames",
            "link_score_used",
        ])

    used_profiles: set[str] = set()
    paths: list[dict] = []
    path_num = 1

    for _, row in df.iterrows():
        a = str(row["profile_id_a"])
        b = str(row["profile_id_b"])

        if a in used_profiles or b in used_profiles:
            continue

        path_id = f"candidate_path_{path_num:03d}"
        path_num += 1

        paths.append({
            "candidate_path_id": path_id,
            "step_num": 1,
            "profile_id": a,
            "camera": row["camera_a"],
            "day": row["day_a"],
            "time_bucket": row["time_bucket_a"],
            "zone_label": row["zone_a"],
            "size_label": row["size_a"],
            "detection_count": row["detection_count_a"],
            "unique_frames": row["unique_frames_a"],
            "link_score_used": row["candidate_link_score"],
        })

        paths.append({
            "candidate_path_id": path_id,
            "step_num": 2,
            "profile_id": b,
            "camera": row["camera_b"],
            "day": row["day_b"],
            "time_bucket": row["time_bucket_b"],
            "zone_label": row["zone_b"],
            "size_label": row["size_b"],
            "detection_count": row["detection_count_b"],
            "unique_frames": row["unique_frames_b"],
            "link_score_used": row["candidate_link_score"],
        })

        used_profiles.add(a)
        used_profiles.add(b)

    out = pd.DataFrame(paths)
    if not out.empty:
        out = out.sort_values(["candidate_path_id", "step_num"]).reset_index(drop=True)
    return out


def build_path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame(columns=[
            "candidate_path_id",
            "day",
            "camera_sequence",
            "time_sequence",
            "path_length",
            "path_score",
        ])

    grouped_rows = []

    for path_id, group in paths.groupby("candidate_path_id"):
        group = group.sort_values("step_num")
        grouped_rows.append({
            "candidate_path_id": path_id,
            "day": group["day"].iloc[0],
            "camera_sequence": " -> ".join(group["camera"].astype(str)),
            "time_sequence": " -> ".join(group["time_bucket"].astype(str)),
            "path_length": len(group),
            "path_score": group["link_score_used"].max(),
        })

    summary = pd.DataFrame(grouped_rows).sort_values(
        ["path_score", "candidate_path_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build simple candidate cross-camera paths from candidate profile links."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/summaries/candidate_profile_links.csv"),
        help="Path to candidate_profile_links.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/summaries/candidate_paths.csv"),
        help="Path to write candidate_paths.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/summaries/candidate_path_summary.csv"),
        help="Path to write candidate_path_summary.csv",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.80,
        help="Minimum candidate_link_score to keep a link for path building.",
    )
    args = parser.parse_args()

    print(f"[INFO] Reading links from {args.input}")
    links = load_links(args.input)

    print(f"[INFO] Loaded {len(links)} raw links")
    links = filter_links(links, min_score=args.min_score)
    print(f"[INFO] Links after filtering: {len(links)}")

    paths = build_paths(links)
    summary = build_path_summary(paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    paths.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)

    num_paths = paths["candidate_path_id"].nunique() if not paths.empty else 0
    print(f"[OK] Wrote {args.output}")
    print(f"[OK] Wrote {args.summary_output}")
    print(f"[INFO] Built {num_paths} candidate paths")


if __name__ == "__main__":
    main()