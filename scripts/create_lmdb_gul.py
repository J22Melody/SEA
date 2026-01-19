"""Create an LMDB file for a given dataset of RGB, flow or pose.
"""
import io
import os
import sys
import json
import shutil
import socket
import tarfile
import argparse
import datetime
import multiprocessing as mp
from typing import Dict, List
from pathlib import Path

import lmdb
import tqdm
import numpy as np
from PIL import Image
from beartype import beartype
from yaspi.yaspi import Yaspi
from zsvision.zs_utils import BlockTimer, memcache, load_json_config
from zsvision.zs_multiproc import starmap_with_kwargs

sys.path.insert(0, "/users/liliane/project_bsl/code/bsltrain")
from exp.featurize import memory_summary, get_video_id_parser


@beartype
def load_all_im_buffers(
    video_idx: int,
    video_id: str,
    limit_frames: int,
    total: int,
    processes: int,
    expected_num_frames: int,
    mute: bool,
    archive_path: Path,
    frame_suffix: str,
    tolerance: int = 1,
    progress_markers: int = 10,
):
    progress_interval = int(max(total, progress_markers) / progress_markers)
    if processes > 1 and video_idx % progress_interval == 0:
        pct = progress_markers * video_idx / total
        print(f"processing {video_idx}/{total} [] [{pct:.1f}%] {video_id}")
        memory_summary()

    if archive_path.suffix == ".tar":
        im_buffers = []
        with tarfile.open(archive_path) as tf:
            members = tf.getmembers()
            im_names = sorted(
                [x.path for x in members if x.path.endswith(frame_suffix)]
            )
            assert (
                abs(len(im_names) - expected_num_frames) <= tolerance
            ), f"Expected {expected_num_frames} frames, found {len(im_names)}"
            if limit_frames:
                im_names = im_names[:limit_frames]
            for im_name in tqdm.tqdm(im_names):
                with BlockTimer("reading", precise=True, mute=mute):
                    im_buffer = tf.extractfile(tf.getmember(im_name)).read()
                im_buffers.append(im_buffer)
    elif archive_path.suffix == ".mat":
        feats = memcache.__wrapped__(archive_path, verbose=False)["preds"]
        im_buffers = feats.astype(np.float16)
        im_names = [f"./{idx + 1:07d}.jpg" for idx in range(len(im_buffers))]
    else:
        raise ValueError(f"Unknown archive suffix: {archive_path.suffix}")

    return {"im_names": im_names, "im_buffers": im_buffers}


@beartype
def get_archive_path(
    archive_dir: Path, archive_type: str, dataset: str, video_id: str,
) -> Path:
    """Construct a path to the archive containing the raw data that we wish
    to insert into an LMDB. The archive will be either a .tar file or a .mat file.

    Args:
        archive_dir: the parent directory containing the archives
    """
    if archive_type == "tar":
        if dataset == "bbcsl_raw":
            fname = f"{video_id}/signhd.tar"
        elif dataset in {"bobsl", "How2Sign"}:
            fname = f"{video_id}.tar"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    elif archive_type == "mat":
        if dataset in {"bbcsl_raw", "bobsl", "How2Sign"}:
            fname = f"{video_id}/features.mat"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    return archive_dir / fname


@beartype
def construct_lmdb(
    video_ids: List[str],
    max_videos_per_chunk: int,
    archive_dir: Path,
    archive_type: str,
    num_frames_per_video: Dict[str, int],
    lmdb_path: Path,
    limit_frames: int,
    frame_suffix: str,
    dataset: str,
    processes: int,
    tolerance: int,
    profile: bool,
    verbose: bool,
    refresh: bool,
    append_to_lmdb: bool,
):
    if lmdb_path.exists():
        if append_to_lmdb:
            print(f"Found existing lmdb at {lmdb_path}, appending to it")
        elif not refresh:
            print(f"Found existing lmdb at {lmdb_path}, skipping")
            return
        else:
            print(f"Removing existing lmdb at {lmdb_path}")
            shutil.rmtree(str(lmdb_path))

    lmdb_path.parent.mkdir(exist_ok=True, parents=True)

    # We use the size occupied by the decoded images as an upper bound on the possible
    # storage required by LMDB, since there seems to be no downside to setting a value
    # that is too high https://github.com/NVIDIA/DIGITS/issues/206
    total_lmdb_frames = sum(num_frames_per_video.values())
    map_size = int(total_lmdb_frames * (256 * 256 * 3))
    env = lmdb.open(str(lmdb_path), map_size=map_size, readonly=False,)

    num_chunks = int(np.ceil(len(video_ids) / max_videos_per_chunk))
    video_id_chunks = np.array_split(video_ids, num_chunks)

    with env.begin(write=True) as txn:

        for chunk_idx, video_id_chunk in tqdm.tqdm(enumerate(video_id_chunks)):
            if verbose:
                print(f"\nProcessing chunk {chunk_idx}/{len(video_id_chunks)}")

            kwarg_list = []
            for video_idx, video_id in tqdm.tqdm(enumerate(video_id_chunk)):
                archive_path = get_archive_path(
                    dataset=dataset,
                    video_id=video_id,
                    archive_dir=archive_dir,
                    archive_type=archive_type,
                )

                kwargs = {
                    "video_idx": video_idx,
                    "processes": processes,
                    "video_id": video_id,
                    "expected_num_frames": num_frames_per_video[video_id],
                    "total": len(video_id_chunk),
                    "mute": not profile,
                    "limit_frames": limit_frames,
                    "frame_suffix": frame_suffix,
                    "archive_path": archive_path,
                    "tolerance": tolerance,
                }
                kwarg_list.append(kwargs)

            func = load_all_im_buffers

            if processes > 1:
                with mp.Pool(processes=processes) as pool:
                    video_frame_buffers = starmap_with_kwargs(
                        pool=pool, func=func, kwargs_iter=kwarg_list,
                    )
            else:
                video_frame_buffers = [
                    func(**kwargs) for kwargs in tqdm.tqdm(kwarg_list)
                ]

            memory_summary()
            for kwargs, buffers in zip(kwarg_list, video_frame_buffers):
                video_id = kwargs["video_id"]
                for im_name, im_buffer in zip(
                    buffers["im_names"], buffers["im_buffers"]
                ):

                    # To enable consistent lmdb key schemes, we account for the fact
                    # that ffmpeg is 1-indexed, whereas openpose is zero indexed
                    if frame_suffix == ".jpg":
                        # Example im_name:
                        # ./0000001.jpg
                        key = str(Path(video_id) / im_name)
                    elif frame_suffix == ".json":
                        # Example im_name:
                        # ./5213407827313563421_000000000000_keypoints.json
                        zero_indexed_frame_num = int(im_name.split("_")[1])
                        one_indexed_frame_num = zero_indexed_frame_num + 1
                        ffmpeg_style_im_name = f"{one_indexed_frame_num:07d}.json"
                        key = str(Path(video_id) / ffmpeg_style_im_name)
                    elif archive_type == "mat" and frame_suffix == ".np":
                        key = str(Path(video_id) / Path(im_name).with_suffix(".np"))
                    else:
                        raise ValueError(f"Unknown frame_suffix: {frame_suffix}")

                    with BlockTimer("writing", precise=True, mute=not profile):
                        txn.put(key.encode("ascii"), im_buffer)


@beartype
def validate_lmdb_folder(
    lmdb_path: Path, video_ids: List[str], frame_suffix: str,
):
    # check we can still read the data (validate the first frame of each video)
    env = lmdb.open(str(lmdb_path), readonly=True)

    with env.begin() as txn:
        print(f"Validating {lmdb_path} for {len(video_ids)} video ids")
        for ii, video_id in tqdm.tqdm(enumerate(video_ids)):
            if frame_suffix == ".jpg":
                im_name = "0000001.jpg"
                key = f"{Path(video_id) / im_name}".encode("ascii")
                data = txn.get(key)
                im = Image.open(io.BytesIO(data))
                print(f"{ii}/{len(video_ids)} Read {im_name} with shape {im.size}")

            elif frame_suffix == ".json":
                pose_file = "0000001.json"
                key = f"{Path(video_id) / pose_file}".encode("ascii")
                data = txn.get(key)
                pose = json.loads(data)
                print(f"{ii}/{len(video_ids)} Read {pose_file} with keys {pose.keys()}")

            elif frame_suffix == ".np":
                pose_file = "0000001.np"
                key = f"{Path(video_id) / pose_file}".encode("ascii")
                data = np.frombuffer(txn.get(key), dtype=np.float16)
                expected_shape = (1024,)
                assert (
                    data.shape == expected_shape
                ), f"Expected shape {expected_shape}, but found {data.shape}"
            else:
                raise ValueError(f"Unknown frame_suffix: {frame_suffix}")


@beartype
def validate_num_frames(
    video_idx: int,
    processes: int,
    frame_suffix: str,
    archive_path: Path,
    archive_type: str,
    total: int,
    video_id: str,
    expected_frames: int,
    tolerance: int = 1,
    progress_markers: int = 100,
) -> Dict:
    progress_interval = int(max(total, progress_markers) / progress_markers)
    if processes > 1 and video_idx % progress_interval == 0:
        pct = progress_markers * video_idx / total
        print(f"processing {video_idx}/{total} [{pct:.1f}%] {video_id}")
        memory_summary()

    issue = {}
    if archive_type == "tar":
        try:
            with tarfile.open(archive_path) as tf:
                members = tf.getmembers()
            im_names = sorted(
                [x.path for x in members if x.path.endswith(frame_suffix)]
            )
            if abs(len(im_names) - expected_frames) > tolerance:
                issue = {
                    "video_id": video_id,
                    "expected": expected_frames,
                    "found": len(im_names),
                }
        except tarfile.ReadError:
            issue = {
                "video_id": video_id,
                "expected": expected_frames,
                "found": -1,
            }
    elif archive_type == "mat":
        # load features silently, without caching
        feats = memcache.__wrapped__(archive_path, verbose=False)["preds"]

        # All features are currently extracted with a 16 frame I3D model with an embedding
        # size of 1024
        expected_feat_dim = 1024
        clip_sz = 16

        # validate the number of features and dimensionality
        num_feats, feat_dim = feats.shape
        assert (
            feat_dim == expected_feat_dim
        ), f"Expected feat dim {expected_feat_dim} but found {feat_dim}"
        # Note: we account for the fact that we do not zero pad the input video
        expected_feats = expected_frames - clip_sz + 1
        num_feats = feats.shape[0]

        if abs(num_feats - expected_feats) > tolerance:
            issue = {
                "video_id": video_id,
                "expected": expected_feats,
                "found": num_feats,
            }
    return issue


@beartype
def validate_archives(
    video_ids: List[str],
    dataset: str,
    frame_suffix: str,
    failure_dir: Path,
    archive_dir: Path,
    archive_type: str,
    num_frames_per_video: Dict[str, int],
    processes: int,
    tolerance: int,
):
    timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    failure_path = failure_dir / f"{timestamp}.txt"

    kwarg_list = []
    for idx, video_id in tqdm.tqdm(enumerate(video_ids)):
        archive_path = get_archive_path(
            dataset=dataset,
            video_id=video_id,
            archive_dir=archive_dir,
            archive_type=archive_type,
        )
        kwargs = {
            "video_idx": idx,
            "video_id": video_id,
            "processes": processes,
            "tolerance": tolerance,
            "expected_frames": num_frames_per_video[video_id],
            "frame_suffix": frame_suffix,
            "archive_path": archive_path,
            "archive_type": archive_type,
            "total": len(video_ids),
        }
        kwarg_list.append(kwargs)

    func = validate_num_frames
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            issues = starmap_with_kwargs(pool=pool, func=func, kwargs_iter=kwarg_list)
    else:
        issues = [func(**kwargs) for kwargs in tqdm.tqdm(kwarg_list)]

    failed = [x for x in issues if x]
    if failed:
        print(f"Writing {len(failed)} failures to {failure_path}")
        failure_path.parent.mkdir(exist_ok=True, parents=True)
        with open(failure_path, "w") as f:
            for failure in failed:
                f.write(
                    f"{failure['video_id']},{failure['expected']},{failure['found']}\n"
                )
    else:
        print(f"No issues were found among {len(video_ids)}")


@beartype
def get_num_frames_for_each_video(
    info_pkl: Path, video_ids: List[str],
) -> Dict[str, int]:

    num_frames = {}
    info = memcache(info_pkl)

    for video_id in video_ids:
        keep = [video_id in name for name in info["videos"]["name"]]
        assert sum(keep) == 1, f"Expected a match for {video_id}, found {sum(keep)}"
        info_idx = keep.index(True)
        num_frames[video_id] = info["videos"]["videos"]["T"][info_idx]
    return num_frames


@beartype
def purge_invalid_archives(
    failure_path: Path, dataset: str, archive_type: str, archive_dir: Path,
):
    with open(failure_path, "r") as f:
        failures = f.read().splitlines()

    tar_paths_to_purge = []
    for failure in failures:
        tokens = failure.split(",")
        assert len(tokens) == 3, (
            "Expected failure format to be csv of the form "
            "`<video_id>,<num found frames>,<num expected frames>` "
            f"but instead found {len(tokens)}"
        )
        video_id = tokens[0]
        tar_path = get_archive_path(
            dataset=dataset,
            video_id=video_id,
            archive_dir=archive_dir,
            archive_type=archive_type,
        )
        if tar_path.exists():
            print(f"Found tar at {tar_path}, adding to removal queue")
            tar_paths_to_purge.append(tar_path)

    for ii, tar_path in enumerate(tar_paths_to_purge):
        print(f"Purging {ii}/{len(tar_paths_to_purge)} {tar_path}")
        tar_path.unlink()


@beartype
def parse_args() -> argparse.Namespace:
    # pylint: disable=line-too-long
    # flake8: noqa: E501
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_list", required=True, type=Path)
    parser.add_argument("--config", type=Path, default="misc/lmdb_configs.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--max_videos_per_chunk", type=int, default=15)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--task",
        choices=[
            "create_lmdb",
            "validate_archives",
            "purge_invalid_tars",
            "validate_lmdb",
        ],
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=[
            "rgb",
            "flow",
            "pose",
            "feats_spottings4_0.7_0-32_mouthings_0.5_last20_poseinit_050",
            "rgb_anon",
            "rgb_no_blur_reencoded",
            "feats_c8889_m8_d8_pretM8D8-12_pret5K_LR3",
            "rgb_right_hand_crop",
            "rgb_left_hand_crop",
        ],
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--append_to_lmdb", action="store_true")
    parser.add_argument(
        "--worker_id",
        default=0,
        type=int,
        help="This argument is used only for yaspi LMDB construction",
    )
    parser.add_argument(
        "--video_list_to_append_to",
        type=Path,
        help="only used when appending to an existing LMDB",
    )
    parser.add_argument(
        "--yaspi_defaults_path", default="misc/yaspi_lmdb_construction_defaults.json"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--failure_path", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--yaspify", action="store_true")
    parser.add_argument("--limit_frames", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if socket.gethostname().endswith("cluster") and args.slurm:
        os.system(str(Path.home() / "configure_tmp_data.sh"))
        print(f"Configured disk for worker: {args.worker_id}")

    config = load_json_config(args.config)[args.dataset]

    with open(args.video_list, "r") as f:
        video_paths = [Path(config["root_dir"]) / x for x in f.read().splitlines()]
    video_id_parser = get_video_id_parser(dataset=args.dataset)
    video_ids = [video_id_parser(x) for x in video_paths]

    if args.dataset == "bbcsl_raw":
        video_ids = [x.replace("--", "/") for x in video_ids]

    # use consistent naming convention when appending to an existing lmdb
    if args.append_to_lmdb:
        assert (
            args.video_list_to_append_to is not None
        ), "--video_list_to_append_to must be supplied"
        stem = args.video_list_to_append_to.stem
        with open(args.video_list_to_append_to, "r") as f:
            orig_paths = [Path(config["root_dir"]) / x for x in f.read().splitlines()]
        original_video_ids = [video_id_parser(x).replace("--", "/") for x in orig_paths]
        video_ids_used_for_meta = video_ids + original_video_ids
    else:
        stem = args.video_list.stem
        video_ids_used_for_meta = video_ids

    lmdb_path = Path(config["lmdb_dir"]) / f"lmdb-{args.modality}-{stem}"

    if args.limit:
        lmdb_path = lmdb_path.parent / f"{lmdb_path.stem}-limit-{args.limit}"

    num_frames_per_video = get_num_frames_for_each_video(
        info_pkl=Path(config["info_pkl"]), video_ids=video_ids_used_for_meta,
    )
    if args.limit:
        video_ids = video_ids[: args.limit]

    #  the rgb frames, flow and pose inputs are all stored in tar files, whose contents
    # correspond to video frames
    if args.modality in {
        "rgb",
        "flow",
        "pose",
        "rgb_anon",
        "rgb_no_blur_reencoded",
        "rgb_right_hand_crop",
        "rgb_left_hand_crop",
    }:
        archive_type = "tar"
        archive_dir = Path(config[f"{args.modality}_frame_tar_dir"])
        if args.modality in {
            "rgb",
            "flow",
            "rgb_anon",
            "rgb_no_blur_reencoded",
            "rgb_right_hand_crop",
            "rgb_left_hand_crop",
        }:
            frame_suffix = ".jpg"
        elif args.modality == "pose":
            frame_suffix = ".json"
    #  features are stored in .mat files, where the number of features depends on the stride
    # used for frame extraction
    elif args.modality.startswith("feats_"):
        archive_dir = Path(config[f"{args.modality}_dir"])
        frame_suffix = ".np"
        archive_type = "mat"
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    if args.task == "validate_archives":
        validate_archives(
            video_ids=video_ids,
            processes=args.processes,
            failure_dir=Path(config["failure_dir"]),
            dataset=args.dataset,
            archive_dir=archive_dir,
            archive_type=archive_type,
            frame_suffix=frame_suffix,
            tolerance=args.tolerance,
            num_frames_per_video=num_frames_per_video,
        )
    elif args.task == "purge_invalid_tars":
        purge_invalid_archives(
            failure_path=args.failure_path,
            dataset=args.dataset,
            archive_dir=archive_dir,
            archive_type=archive_type,
        )
    elif args.task == "create_lmdb":
        if args.yaspify:
            with open(args.yaspi_defaults_path, "r") as f:
                yaspi_defaults = json.load(f)
            cmd_args = sys.argv
            cmd_args.remove("--yaspify")
            job = Yaspi(
                cmd=f"python {' '.join(cmd_args)}",
                job_name=f"create_lmdb-{args.dataset}",
                job_queue=None,
                gpus_per_task=0,
                job_array_size=1,
                **yaspi_defaults,
            )
            job.submit(watch=True, conserve_resources=5)
        else:
            construct_lmdb(
                video_ids=video_ids,
                dataset=args.dataset,
                tolerance=args.tolerance,
                max_videos_per_chunk=args.max_videos_per_chunk,
                num_frames_per_video=num_frames_per_video,
                archive_dir=archive_dir,
                archive_type=archive_type,
                limit_frames=args.limit_frames,
                frame_suffix=frame_suffix,
                processes=args.processes,
                refresh=args.refresh,
                profile=args.profile,
                verbose=args.verbose,
                lmdb_path=lmdb_path,
                append_to_lmdb=args.append_to_lmdb,
            )
    elif args.task == "validate_lmdb":
        validate_lmdb_folder(
            lmdb_path=lmdb_path, video_ids=video_ids, frame_suffix=frame_suffix,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()