from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


DAY_TO_NUM = {
    "day1": 1,
    "day2": 2,
    "day3": 3,
    "day4": 4,
    "day5": 5,
    "day6": 6,
    "day7": 7,
}


SIZE_TO_NUM = {
    "small": 0,
    "medium": 1,
    "large": 2,
    "xlarge": 3,
    "xxlarge": 4,
}


def load_profiles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Profiles file is empty: {path}")

    required = {
        "profile_id",
        "camera",
        "day",
        "zone_label",
        "size_label",
        "time_bucket_label",
        "detection_count",
        "unique_frames",
        "avg_bbox_area",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def parse_time_bucket_label(label: str) -> int:
    """
    Example: '18:10-18:19' -> 1090 minutes (midpoint approx 18:15)
    """
    start, end = label.split("-")
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    return (start_min + end_min) // 2


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_num"] = df["day"].map(DAY_TO_NUM)
    df["size_num"] = df["size_label"].map(SIZE_TO_NUM).fillna(1)
    df["time_mid_min"] = df["time_bucket_label"].apply(parse_time_bucket_label)

    max_det = max(float(df["detection_count"].max()), 1.0)
    max_frames = max(float(df["unique_frames"].max()), 1.0)
    max_area = max(float(df["avg_bbox_area"].max()), 1.0)

    df["det_norm"] = df["detection_count"] / max_det
    df["frames_norm"] = df["unique_frames"] / max_frames
    df["area_norm"] = df["avg_bbox_area"] / max_area

    return df


def size_similarity(a: float, b: float) -> float:
    diff = abs(a - b)
    return max(0.0, 1.0 - diff / 4.0)


def time_similarity(a: int, b: int, max_diff_min: int = 30) -> float:
    diff = abs(a - b)
    return max(0.0, 1.0 - diff / max_diff_min)


def day_similarity(a: float | int, b: float | int) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0
    diff = abs(int(a) - int(b))
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.6
    return 0.0


def evidence_strength(det_norm: float, frames_norm: float) -> float:
    return 0.5 * det_norm + 0.5 * frames_norm


def score_pair(row_a: pd.Series, row_b: pd.Series, allow_same_camera: bool) -> dict | None:
    if row_a["profile_id"] == row_b["profile_id"]:
        return None

    # avoid duplicate pair ordering
    if row_a["profile_id"] > row_b["profile_id"]:
        return None

    same_camera = row_a["camera"] == row_b["camera"]
    if same_camera and not allow_same_camera:
        return None

    t_sim = time_similarity(int(row_a["time_mid_min"]), int(row_b["time_mid_min"]))
    if t_sim <= 0:
        return None

    s_sim = size_similarity(float(row_a["size_num"]), float(row_b["size_num"]))
    d_sim = day_similarity(row_a["day_num"], row_b["day_num"])
    e_a = evidence_strength(float(row_a["det_norm"]), float(row_a["frames_norm"]))
    e_b = evidence_strength(float(row_b["det_norm"]), float(row_b["frames_norm"]))
    evidence = (e_a + e_b) / 2.0

    # same-day different-camera links are the strongest for your demo
    same_day = row_a["day"] == row_b["day"]
    cross_camera = row_a["camera"] != row_b["camera"]

    same_day_bonus = 0.15 if same_day else 0.0
    cross_camera_bonus = 0.15 if cross_camera else 0.0

    score = (
        0.35 * t_sim +
        0.20 * s_sim +
        0.15 * d_sim +
        0.15 * evidence +
        same_day_bonus +
        cross_camera_bonus
    )

    score = min(score, 1.0)

    if score >= 0.80:
        strength = "strong"
    elif score >= 0.65:
        strength = "moderate"
    elif score >= 0.50:
        strength = "weak"
    else:
        return None

    return {
        "profile_id_a": row_a["profile_id"],
        "camera_a": row_a["camera"],
        "day_a": row_a["day"],
        "time_bucket_a": row_a["time_bucket_label"],
        "zone_a": row_a["zone_label"],
        "size_a": row_a["size_label"],
        "detection_count_a": row_a["detection_count"],
        "unique_frames_a": row_a["unique_frames"],
        "profile_id_b": row_b["profile_id"],
        "camera_b": row_b["camera"],
        "day_b": row_b["day"],
        "time_bucket_b": row_b["time_bucket_label"],
        "zone_b": row_b["zone_label"],
        "size_b": row_b["size_label"],
        "detection_count_b": row_b["detection_count"],
        "unique_frames_b": row_b["unique_frames"],
        "same_camera": same_camera,
        "cross_camera": cross_camera,
        "same_day": same_day,
        "time_similarity": round(t_sim, 4),
        "size_similarity": round(s_sim, 4),
        "day_similarity": round(d_sim, 4),
        "evidence_strength": round(evidence, 4),
        "candidate_link_score": round(score, 4),
        "link_strength": strength,
        "link_explanation": make_link_explanation(
            row_a, row_b, score, strength, same_day, cross_camera
        ),
    }


def make_link_explanation(
    row_a: pd.Series,
    row_b: pd.Series,
    score: float,
    strength: str,
    same_day: bool,
    cross_camera: bool,
) -> str:
    parts = []

    if cross_camera:
        parts.append("cross-camera")
    else:
        parts.append("same-camera")

    if same_day:
        parts.append("same-day")
    else:
        parts.append("cross-day")

    parts.append(f"similar time windows ({row_a['time_bucket_label']} vs {row_b['time_bucket_label']})")
    parts.append(f"similar size groups ({row_a['size_label']} vs {row_b['size_label']})")

    return (
        f"{strength.capitalize()} candidate link ({score:.2f}): "
        + ", ".join(parts)
    )


def build_links(df: pd.DataFrame, allow_same_camera: bool) -> pd.DataFrame:
    rows = []
    records = list(df.to_dict(orient="records"))

    start_time = time.time()
    total_profiles = len(records)
    total_pairs = total_profiles * (total_profiles - 1) // 2
    
    processed_pairs = 0
    last_print = 0
    
    for i in range(total_profiles):
        row_a = pd.Series(records[i])
    
        for j in range(i + 1, total_profiles):
            row_b = pd.Series(records[j])
    
            processed_pairs += 1
    
            if processed_pairs - last_print >= 100_000:
                elapsed = time.time() - start_time
                rate = processed_pairs / elapsed if elapsed > 0 else 0
    
                remaining = total_pairs - processed_pairs
                eta_seconds = remaining / rate if rate > 0 else 0
    
                eta_minutes = eta_seconds / 60
                percent = (processed_pairs / total_pairs) * 100
    
                print(
                    f"[PROGRESS] {percent:.2f}% | "
                    f"{processed_pairs:,}/{total_pairs:,} pairs | "
                    f"ETA: {eta_minutes:.1f} min"
                )
    
                last_print = processed_pairs
    
            pair = score_pair(row_a, row_b, allow_same_camera=allow_same_camera)
            if pair is not None:
                rows.append(pair)
    
    if not rows:
        return pd.DataFrame()

    links = pd.DataFrame(rows)
    links = links.sort_values(
        ["candidate_link_score", "cross_camera", "same_day"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return links


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_candidate_groups(profiles: pd.DataFrame, links: pd.DataFrame, min_group_score: float) -> pd.DataFrame:
    if links.empty:
        return pd.DataFrame(columns=[
            "candidate_group_id",
            "profile_id",
            "camera",
            "day",
            "time_bucket_label",
            "size_label",
            "detection_count",
            "unique_frames",
        ])

    uf = UnionFind(profiles["profile_id"].tolist())

    strong_enough = links[links["candidate_link_score"] >= min_group_score]
    for _, row in strong_enough.iterrows():
        uf.union(row["profile_id_a"], row["profile_id_b"])

    group_map = {pid: uf.find(pid) for pid in profiles["profile_id"]}
    root_to_group = {}
    next_idx = 1
    for root in sorted(set(group_map.values())):
        root_to_group[root] = f"candidate_group_{next_idx:03d}"
        next_idx += 1

    out = profiles.copy()
    out["candidate_group_id"] = out["profile_id"].map(lambda pid: root_to_group[group_map[pid]])

    counts = out["candidate_group_id"].value_counts()
    out = out[out["candidate_group_id"].map(counts) > 1].copy()

    if out.empty:
        return pd.DataFrame(columns=[
            "candidate_group_id",
            "profile_id",
            "camera",
            "day",
            "time_bucket_label",
            "size_label",
            "detection_count",
            "unique_frames",
        ])

    out = out[
        [
            "candidate_group_id",
            "profile_id",
            "camera",
            "day",
            "time_bucket_label",
            "size_label",
            "detection_count",
            "unique_frames",
        ]
    ].sort_values(["candidate_group_id", "camera", "day", "time_bucket_label"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link pseudonymous profiles into candidate cross-camera/cross-day groups."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/summaries/pseudonymous_profiles.csv"),
        help="Path to pseudonymous_profiles.csv",
    )
    parser.add_argument(
        "--links-output",
        type=Path,
        default=Path("data/summaries/candidate_profile_links.csv"),
        help="Output CSV for pairwise candidate links",
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=Path("data/summaries/candidate_profile_groups.csv"),
        help="Output CSV for grouped candidate profiles",
    )
    parser.add_argument(
        "--allow-same-camera",
        action="store_true",
        help="Also allow same-camera cross-day links. Default only outputs cross-camera links.",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.65,
        help="Minimum candidate_link_score to merge profiles into the same candidate group.",
    )
    parser.add_argument(
        "--min-detection-count",
        type=int,
        default=3,
        help="Minimum detection_count required to keep a profile before linking.",
    )
    parser.add_argument(
        "--min-unique-frames",
        type=int,
        default=2,
        help="Minimum unique_frames required to keep a profile before linking.",
    )
    args = parser.parse_args()

    print(f"[INFO] Reading profiles from {args.input}")
    profiles = load_profiles(args.input)
    
    print(f"[INFO] Loaded {len(profiles)} raw profiles")
    
    # Filter out weak / one-off profiles before linking
    profiles = profiles[
        (profiles["detection_count"] >= args.min_detection_count) &
        (profiles["unique_frames"] >= args.min_unique_frames)
    ].copy()
    
    print(f"[INFO] Profiles after filtering: {len(profiles)}")
    
    if profiles.empty:
        print("[INFO] No profiles remain after filtering.")
        args.links_output.parent.mkdir(parents=True, exist_ok=True)
        args.groups_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.links_output, index=False)
        pd.DataFrame().to_csv(args.groups_output, index=False)
        return
    
    profiles = prepare_features(profiles)
    
    print("[INFO] Building candidate links")
    n = len(profiles)
    total_pairs = n * (n - 1) // 2
    print(f"[INFO] About to compare {n:,} profiles ({total_pairs:,} pairs)")
    links = build_links(profiles, allow_same_camera=args.allow_same_camera)

    args.links_output.parent.mkdir(parents=True, exist_ok=True)
    args.groups_output.parent.mkdir(parents=True, exist_ok=True)

    if links.empty:
        print("[INFO] No candidate links found with the current thresholds.")
        pd.DataFrame().to_csv(args.links_output, index=False)
        pd.DataFrame().to_csv(args.groups_output, index=False)
        return

    if not args.allow_same_camera:
        links = links[links["cross_camera"]].copy()

    links = links.sort_values(
        ["candidate_link_score", "cross_camera", "same_day"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    links.to_csv(args.links_output, index=False)
    print(f"[OK] Wrote {args.links_output}")

    print("[INFO] Building candidate groups")
    groups = build_candidate_groups(profiles, links, min_group_score=args.group_threshold)
    groups.to_csv(args.groups_output, index=False)
    print(f"[OK] Wrote {args.groups_output}")

    print("[DONE] Candidate linking complete.")


if __name__ == "__main__":
    main()