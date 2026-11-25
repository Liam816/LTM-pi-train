"""Repack the original RoboMME LeRobot v3 dataset as v2.0 for the pinned v2.1 reader."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


VIEWS = ("image", "wrist_image")
EXPORT_NAMES = {
    "observation.images.front": "image", "observation.images.wrist": "wrist_image",
    "observation.state": "state", "action": "actions", "is_video_demo": "is_demo",
}
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_path(root, template, chunks_size, episode, **kwargs):
    index = episode["episode_index"]
    return root / template.format(episode_chunk=index // chunks_size, episode_index=index, **kwargs)


def shard_groups(episodes, prefix, path_template, **kwargs):
    groups = defaultdict(list)
    for episode in episodes:
        path = path_template.format(
            chunk_index=episode[f"{prefix}/chunk_index"],
            file_index=episode[f"{prefix}/file_index"],
            **kwargs,
        )
        groups[path].append(episode)
    return groups


def repack_tables(source, output, info, episodes, tasks):
    metadata = []
    for relative, group in shard_groups(episodes, "data", info["data_path"]).items():
        shard = pq.read_table(source / relative)
        first_index = int(shard["index"][0].as_py())
        for episode in group:
            index = episode["episode_index"]
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            episode_table = shard.slice(start - first_index, end - start)
            episode_table = episode_table.rename_columns([EXPORT_NAMES.get(k, k) for k in episode_table.column_names])
            frames = episode_table["frame_index"].to_numpy()
            global_rows = episode_table["index"].to_numpy()
            timestamps = episode_table["timestamp"].to_numpy()
            if (
                len(frames) != episode["length"]
                or not np.all(episode_table["episode_index"].to_numpy() == index)
                or not np.array_equal(frames, np.arange(episode["length"]))
                or not np.array_equal(global_rows, np.arange(start, end))
                or not np.allclose(timestamps, frames / info["fps"], atol=2e-5, rtol=0)
            ):
                raise ValueError(f"Frame order or length mismatch in episode {index}")
            demo = episode_table["is_demo"].to_numpy().astype(bool)
            exec_start = int(demo.sum())
            if exec_start >= len(frames) or not np.array_equal(demo, frames < exec_start):
                raise ValueError(f"Invalid demo/execution prefix in episode {index}")
            if "exec_start_idx" in episode_table.column_names:
                if not np.all(episode_table["exec_start_idx"].to_numpy() == exec_start):
                    raise ValueError(f"Inconsistent execution boundary in episode {index}")
            else:
                episode_table = episode_table.append_column("exec_start_idx", pa.array(np.full(len(frames), exec_start, np.int64)))
            for key in ("state", "actions"):
                values = pc.cast(episode_table[key], pa.list_(pa.float32(), 8))
                episode_table = episode_table.set_column(episode_table.schema.get_field_index(key), key, values)
            episode_table = episode_table.replace_schema_metadata(None)
            path = episode_path(output, DATA_PATH, info["chunks_size"], episode)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, path)
            instruction_ids = sorted(set(episode_table["task_index"].to_pylist()))
            metadata.append(
                {"episode_index": index, "length": len(frames), "tasks": [tasks[i] for i in instruction_ids]}
            )
    return sorted(metadata, key=lambda item: item["episode_index"])


def frame_at(seconds, fps):
    frame = round(seconds * fps)
    if not np.isclose(seconds * fps, frame, atol=1e-4, rtol=0):
        raise ValueError(f"Video timestamp {seconds} is off the {fps} fps frame grid")
    return frame


def copy_episode_video(source_file, destination, start, length, fps):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # MOV's edit list hides pre-roll when an episode begins on a non-keyframe.
    # The video packets are copied without re-encoding either camera.
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror",
        "-ss", f"{start / fps:.6f}", "-i", str(source_file), "-map", "0:v:0",
        "-c:v", "copy", "-t", f"{length / fps:.6f}", str(destination),
    ]
    subprocess.run(command, check=True)
    with av.open(str(destination)) as video:
        stream = video.streams.video[0]
        timeline = sorted(
            frame_at(float(packet.pts * stream.time_base), fps)
            for packet in video.demux(stream)
            if packet.pts is not None and packet.pts >= 0
        )
    if timeline != list(range(length)):
        raise ValueError(f"Video timeline does not match the expected {length} frames: {destination}")


def repack_videos(source, output, info, episodes):
    fps = info["fps"]
    source_views = [key for key in info["features"] if EXPORT_NAMES.get(key, key) in VIEWS]
    if len(source_views) != 2:
        raise ValueError("Expected one front camera and one wrist camera")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for source_view in source_views:
            view = EXPORT_NAMES.get(source_view, source_view)
            prefix = f"videos/{source_view}"
            for relative, group in shard_groups(episodes, prefix, info["video_path"], video_key=source_view).items():
                group.sort(key=lambda e: e[f"{prefix}/from_timestamp"])
                jobs = []
                start = 0
                for episode in group:
                    begin = frame_at(episode[f"{prefix}/from_timestamp"], fps)
                    end = frame_at(episode[f"{prefix}/to_timestamp"], fps)
                    length = episode["length"]
                    if begin != start or end - begin != length:
                        raise ValueError(
                            f"Video span is not contiguous for {relative}, episode {episode['episode_index']}"
                        )
                    destination = episode_path(
                        output, VIDEO_PATH, info["chunks_size"], episode, video_key=view
                    )
                    jobs.append((source / relative, destination, begin, length, fps))
                    start = end
                print(f"{relative}: repacking {len(group)} episodes", flush=True)
                for _ in pool.map(lambda args: copy_episode_video(*args), jobs):
                    pass


def repack(source, destination, limit):
    source = source.resolve()
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("The output directory must be outside the original dataset")
    if destination.exists():
        raise FileExistsError(destination)

    info = json.loads((source / "meta" / "info.json").read_text())
    length = info["total_episodes"]
    if (info["codebase_version"], length, info["total_frames"], info["splits"]) != (
        "v3.0", 1600, 768897, {"train": "0:1600"}
    ) or info["fps"] not in (10, 20):
        raise ValueError("The source must be the original RoboMME v3.0 training split")
    if limit is not None and not 1 <= limit <= length:
        raise ValueError(f"--limit-episodes must be between 1 and {length}")

    rows = pq.read_table(source / "meta" / "episodes").to_pylist()
    rows.sort(key=lambda e: e["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(length)):
        raise ValueError("The source has missing, duplicate, or invalid episode IDs")
    episodes = rows[:limit]
    tasks = pq.read_table(source / "meta" / "tasks.parquet").to_pylist()
    task_map = {task["task_index"]: task["task"] for task in tasks}
    length = len(episodes)
    frames = sum(episode["length"] for episode in episodes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robomme-v2-", dir=destination.parent) as temporary:
        output = Path(temporary) / "dataset"
        meta = output / "meta"
        meta.mkdir(parents=True)
        metadata = repack_tables(source, output, info, episodes, task_map)
        repack_videos(source, output, info, episodes)
        write_jsonl(meta / "tasks.jsonl", tasks)
        write_jsonl(meta / "episodes.jsonl", metadata)
        # The original export has dataset-level stats and no episode-level
        # stats. The v2.1 reader supports the v2.0 global-stat layout.
        raw_stats = json.loads((source / "meta/stats.json").read_text())
        stats = {EXPORT_NAMES.get(key, key): value for key, value in raw_stats.items()}
        if "exec_start_idx" not in stats:
            lengths = np.array([row["length"] for row in metadata])
            boundaries = np.array([
                pq.read_table(episode_path(output, DATA_PATH, info["chunks_size"], row), columns=["exec_start_idx"])
                ["exec_start_idx"][0].as_py() for row in metadata
            ])
            b = np.repeat(boundaries, lengths)
            stats["exec_start_idx"] = dict(min=[int(b.min())], max=[int(b.max())], mean=[float(b.mean())], std=[float(b.std())], count=[len(b)])
        (meta / "stats.json").write_text(json.dumps(stats) + "\n")
        features = {EXPORT_NAMES.get(k, k): v for k, v in info["features"].items()}
        features.setdefault("exec_start_idx", dict(dtype="int64", shape=[1], names=None))
        for view in VIEWS:
            camera = features[view]
            desc = camera["info"]
            camera.update(shape=[desc["video.height"], desc["video.width"], desc["video.channels"]],
                          names=["height", "width", "channel"], fps=info["fps"])
        info.update(
            features=features,
            codebase_version="v2.0",
            data_path=DATA_PATH,
            video_path=VIDEO_PATH,
            splits={"train": f"0:{length}"},
            total_episodes=length,
            total_frames=frames,
            total_tasks=len(tasks),
            total_videos=length * len(VIEWS),
            total_chunks=(length + info["chunks_size"] - 1) // info["chunks_size"],
        )
        (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        (output / "_READY").touch()
        output.rename(destination)
    print(f"{destination}: {length} episodes, {frames} frames, AV1 stream copy", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit-episodes", type=int, help="Use a prefix of episodes for a local smoke test")
    args = parser.parse_args()
    repack(args.source, args.output, args.limit_episodes)


if __name__ == "__main__":
    main()
