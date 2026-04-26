from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAMERA_LABELS = {
    "balcony": "Balcony",
    "bar_stage": "Bar stage",
    "inside_bar": "Inside bar",
    "street_view": "Street view",
}


def load_detections(detections_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(detections_dir.glob("*/*_detections.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No detection CSVs found under {detections_dir}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["camera_label"] = df["camera"].map(CAMERA_LABELS).fillna(df["camera"])
    df["frame_number"] = (
        df["frame_file"]
        .str.extract(r"_(\d+)\.", expand=False)
        .astype(float)
        .astype("Int64")
    )
    df["has_cross_camera_match"] = df["global_match_score"].notna()
    return df


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {path}")


def figure_detection_volume(df: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        df.groupby("camera_label")
        .agg(
            detections=("frame_file", "count"),
            frames=("frame_file", "nunique"),
            tracks=("track_id", "nunique"),
            global_profiles=("global_id", "nunique"),
        )
        .reindex([CAMERA_LABELS[c] for c in CAMERA_LABELS if CAMERA_LABELS[c] in set(df["camera_label"])])
    )

    ax = counts[["detections", "tracks", "global_profiles"]].plot(
        kind="bar",
        figsize=(10, 5.6),
        color=["#3b82f6", "#f59e0b", "#10b981"],
        width=0.78,
    )
    ax.set_title("Observable Identity Signals by Public Camera")
    ax.set_xlabel("Camera view")
    ax.set_ylabel("Count")
    ax.legend(["Person detections", "Per-camera tracks", "Pseudonymous global profiles"])
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8, padding=2)

    save_fig(out_dir / "01_detection_volume_by_camera.png")


def figure_cross_camera_profiles(df: pd.DataFrame, out_dir: Path) -> None:
    profiles = (
        df.groupby("global_id")
        .agg(
            cameras=("camera", "nunique"),
            detections=("frame_file", "count"),
            frames=("frame_file", "nunique"),
            first_frame=("frame_number", "min"),
            last_frame=("frame_number", "max"),
        )
        .reset_index()
    )
    profiles["camera_span"] = profiles["cameras"].clip(upper=4)
    span_counts = profiles["camera_span"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [str(int(x)) for x in span_counts.index],
        span_counts.values,
        color=["#94a3b8", "#6366f1", "#8b5cf6", "#d946ef"][: len(span_counts)],
    )
    ax.set_title("Pseudonymous Profiles Can Span Multiple Public Feeds")
    ax.set_xlabel("Number of distinct camera views linked to one global ID")
    ax.set_ylabel("Number of global IDs")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%d", padding=3)

    save_fig(out_dir / "02_global_ids_by_camera_span.png")


def figure_persistence_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    profiles = (
        df.groupby("global_id")
        .agg(
            detections=("frame_file", "count"),
            cameras=("camera", "nunique"),
            tracks=("track_id", "nunique"),
        )
        .sort_values("detections", ascending=False)
        .head(20)
        .sort_values("detections")
    )

    colors = np.where(profiles["cameras"] > 1, "#dc2626", "#2563eb")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh([f"G{int(g)}" for g in profiles.index], profiles["detections"], color=colors)
    ax.set_title("Most Persistent Pseudonymous Profiles")
    ax.set_xlabel("Total detections across frames")
    ax.set_ylabel("Global ID")
    ax.grid(axis="x", alpha=0.25)

    for bar, cameras in zip(bars, profiles["cameras"]):
        ax.text(
            bar.get_width() + max(profiles["detections"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(cameras)} cam",
            va="center",
            fontsize=8,
        )

    save_fig(out_dir / "03_top_persistent_profiles.png")


def figure_observation_timeline(df: pd.DataFrame, out_dir: Path) -> None:
    top_ids = (
        df.groupby("global_id")["frame_file"]
        .count()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    plot_df = df[df["global_id"].isin(top_ids)].copy()
    plot_df["profile"] = "G" + plot_df["global_id"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    camera_to_y_offset = {
        camera: i * 0.17 for i, camera in enumerate(sorted(plot_df["camera"].unique()))
    }
    y_positions = {profile: i for i, profile in enumerate(sorted(plot_df["profile"].unique()))}
    colors = {
        "balcony": "#2563eb",
        "bar_stage": "#f59e0b",
        "inside_bar": "#10b981",
        "street_view": "#ef4444",
    }

    for camera, group in plot_df.groupby("camera"):
        x = group["frame_number"].astype(float)
        y = group["profile"].map(y_positions).astype(float) + camera_to_y_offset[camera]
        ax.scatter(
            x,
            y,
            s=12,
            alpha=0.7,
            label=CAMERA_LABELS.get(camera, camera),
            color=colors.get(camera, "#64748b"),
        )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_title("Repeated Observations Build Persistent Pseudonymous Profiles")
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Global ID")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncols=2, fontsize=8)

    save_fig(out_dir / "04_profile_observation_timeline.png")


def figure_confidence_and_matches(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df.boxplot(column="confidence", by="camera_label", ax=axes[0], grid=False)
    axes[0].set_title("Detector Confidence by Camera")
    axes[0].set_xlabel("Camera view")
    axes[0].set_ylabel("YOLO confidence")
    axes[0].tick_params(axis="x", rotation=20)

    cross = (
        df.groupby("camera_label")["has_cross_camera_match"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "cross_matches", "count": "detections"})
    )
    cross["share"] = cross["cross_matches"] / cross["detections"]
    bars = axes[1].bar(cross.index, cross["share"], color="#dc2626")
    axes[1].set_title("Share of Detections Linked Across Cameras")
    axes[1].set_xlabel("Camera view")
    axes[1].set_ylabel("Cross-camera linked share")
    axes[1].set_ylim(0, max(cross["share"].max() * 1.25, 0.05))
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(bars, labels=[f"{v:.1%}" for v in cross["share"]], padding=3)

    fig.suptitle("")
    save_fig(out_dir / "05_confidence_and_cross_camera_matches.png")


def make_video_contact_sheet(video_paths: list[Path], out_dir: Path) -> None:
    frames = []
    labels = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open {video_path}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_points = [0, max(total // 2, 0), max(total - 1, 0)]

        for point in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, point)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (360, 203))
            frames.append(frame)
            labels.append(f"{video_path.stem}, frame {point}")

        cap.release()

    if not frames:
        return

    cols = 3
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.1 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, frame, label in zip(axes, frames, labels):
        ax.imshow(frame)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    for ax in axes[len(frames):]:
        ax.axis("off")

    save_fig(out_dir / "06_video_contact_sheet.png")


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    profiles = df.groupby("global_id").agg(
        cameras=("camera", "nunique"),
        detections=("frame_file", "count"),
        frames=("frame_file", "nunique"),
        tracks=("track_id", "nunique"),
    )
    summary = {
        "total_detections": int(len(df)),
        "camera_views": int(df["camera"].nunique()),
        "frames_with_detections": int(df["frame_file"].nunique()),
        "per_camera_tracks": int(df["track_id"].nunique()),
        "global_profiles": int(df["global_id"].nunique()),
        "profiles_seen_in_multiple_cameras": int((profiles["cameras"] > 1).sum()),
        "detections_with_cross_camera_match_score": int(df["global_match_score"].notna().sum()),
    }

    out_path = out_dir / "summary_metrics.csv"
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")

    top_path = out_dir / "top_persistent_profiles.csv"
    profiles.sort_values(["cameras", "detections"], ascending=False).head(25).to_csv(top_path)
    print(f"[OK] Wrote {top_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from detection CSVs.")
    parser.add_argument("--detections-dir", type=Path, default=Path("data/detections"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--videos", nargs="*", type=Path, default=list(Path(".").glob("*.mp4")))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_detections(args.detections_dir)

    figure_detection_volume(df, args.out_dir)
    figure_cross_camera_profiles(df, args.out_dir)
    figure_persistence_distribution(df, args.out_dir)
    figure_observation_timeline(df, args.out_dir)
    figure_confidence_and_matches(df, args.out_dir)
    make_video_contact_sheet(args.videos, args.out_dir)
    write_summary(df, args.out_dir)


if __name__ == "__main__":
    main()
