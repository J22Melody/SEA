#!/usr/bin/env python3
import argparse
import os
import subprocess
from itertools import product
from tqdm import tqdm
import multiprocessing


def process_video(vid, args, model, sign_b, sign_o):
    # Pin this worker process to a dedicated CPU from its allowed set.
    try:
        available_cpus = os.sched_getaffinity(0)
        cpu_list = sorted(available_cpus)
        cpu_id = cpu_list[os.getpid() % len(cpu_list)]
        os.sched_setaffinity(0, {cpu_id})
    except Exception as e:
        print(f"Error setting CPU affinity for video {vid}: {e}")

    # Determine the sub-directory name.
    model_name = model
    if model_name.startswith("model_"):
        model_name = model_name[len("model_"):]
    if model_name.endswith(".pth"):
        model_name = model_name[:-4]
    sub_save_dir = os.path.join(args.save_dir, f"{model_name}_{sign_b}_{sign_o}")
    os.makedirs(sub_save_dir, exist_ok=True)
    
    pose_file = os.path.join(args.pose_dir, f"{vid}.pose")
    elan_file = os.path.join(sub_save_dir, f"{vid}.eaf")

    # Skip processing if output already exists and overwrite is not set.
    if not args.overwrite and os.path.exists(elan_file):
        return f"Skipping {vid} for {model_name}_{sign_b}_{sign_o}: output already exists at {elan_file}"

    # Build the pose_to_segments command using the current combination.
    cmd = (
        f"pose_to_segments --no-pose-link --model={model} "
        f"--pose={pose_file} --elan={elan_file} "
        f"--sign-b-threshold {sign_b} --sign-o-threshold {sign_o}"
    )

    # Check for the video file.
    video_file = os.path.join(args.video_dir, f"{vid}.mp4")
    if os.path.exists(video_file):
        cmd += f" --video=./{vid}.mp4"

    # Check for the automatic subtitles file (.vtt or .srt)
    subtitle_file = None
    for ext in ['.vtt', '.srt']:
        candidate = os.path.join(args.subtitle_dir, f"{vid}{ext}")
        if os.path.exists(candidate):
            subtitle_file = candidate
            break
    if subtitle_file:
        cmd += f" --subtitles={subtitle_file}"

    # Check for the manually corrected subtitles file (.vtt or .srt)
    subtitle_corrected_file = None
    for ext in ['.vtt', '.srt']:
        candidate = os.path.join(args.subtitle_dir_corrected, f"{vid}{ext}")
        if os.path.exists(candidate):
            subtitle_corrected_file = candidate
            break
    if subtitle_corrected_file:
        cmd += f" --subtitles-corrected={subtitle_corrected_file}"

    # Run the command.
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        return f"Error processing video id {vid} for {model_name}_{sign_b}_{sign_o} (return code {result.returncode}): {cmd}"
    return f"Processed {vid} for {model_name}_{sign_b}_{sign_o}"  

def process_task(task):
    vid, model, sign_b, sign_o, args = task
    return process_video(vid, args, model, sign_b, sign_o)


def main():
    parser = argparse.ArgumentParser(
        description="Segment videos based on their pose files and save results."
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        default="/users/zifan/subtitle_align/data/bobsl_align.txt",
        help="Path to text file containing video ids (one per line), or 'all' to auto-discover from pose_dir."
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe_v2_refine_face_complexity_2",
        help="Directory where pose files are stored."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/segmentation",
        help="Directory to store segmentation results."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing feature files if set"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/users/zifan/BOBSL/derivatives/original_videos",
        help="Directory containing original videos."
    )
    parser.add_argument(
        "--subtitle_dir",
        type=str,
        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
        help="Directory containing automatically aligned subtitles."
    )
    parser.add_argument(
        "--subtitle_dir_corrected",
        type=str,
        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
        help="Directory containing manually corrected subtitles."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers to process videos. Default is 1 (sequential processing)."
    )
    parser.add_argument("--model", nargs='+', default=["model_E4s-1.pth"], type=str, help="Path(s) to model file")
    parser.add_argument("--sign-b-threshold", nargs='+', default=[60], type=int, help="Threshold(s) for sign B")
    parser.add_argument("--sign-o-threshold", nargs='+', default=[50], type=int, help="Threshold(s) for sign O")
    args = parser.parse_args()

    # Ensure that the save directory exists.
    os.makedirs(args.save_dir, exist_ok=True)

    # Determine video IDs
    if args.video_ids.lower() == "all":
        # Discover all pose files
        try:
            files = os.listdir(args.pose_dir)
            video_ids = [os.path.splitext(f)[0] for f in files if f.endswith('.pose')]
            print(f"Discovered {len(video_ids)} videos from pose_dir: {video_ids}")
        except Exception as e:
            print(f"Error listing pose_dir '{args.pose_dir}': {e}")
            return
    else:
        # Read video ids from the provided file.
        with open(args.video_ids, "r") as file:
            video_ids = [line.strip() for line in file if line.strip()]

    # Create all combinations of model, sign-b-threshold, and sign-o-threshold.
    combinations = list(product(args.model, args.sign_b_threshold, args.sign_o_threshold))
    
    # Build tasks as (video_id, model, sign_b, sign_o, args) for each video and each combination.
    tasks = []
    for vid in video_ids:
        for combo in combinations:
            tasks.append((vid, combo[0], combo[1], combo[2], args))
    
    # Process tasks
    if args.num_workers > 1:
        with multiprocessing.Pool(args.num_workers) as pool:
            for res in tqdm(pool.imap_unordered(process_task, tasks),
                            total=len(tasks), desc="Processing videos"):
                tqdm.write(res)
    else:
        for task in tqdm(tasks, desc="Processing videos"):
            res = process_task(task)
            tqdm.write(res)

if __name__ == "__main__":
    main()
