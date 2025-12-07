"""Compute subtitle alignment metrics.

This module provides code to evaluate the quality of a set of subtitle alignments (i.e.
how well their start and end times match those of a set of ground-truth subtitle times).

Note: several of the metrics are derived from the following code:
    https://github.com/yabufarha/ms-tcn/blob/
        c6eab71ddd7b4190ffb3bf6f1b57f3517454939b/eval.py#L15

Example usage:
python misc/sub_align/evaluate_sub_alignment.py
(All arguments are now expected to be in the config file loaded by load_opts)
"""
import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"), "subtitle_align"))  # dynamic home path

from pickle import SHORT_BINSTRING
from typing import List, Tuple
from pathlib import Path
import multiprocessing
import warnings
import csv

import tqdm
from statistics import mean, median
import numpy as np
import webvtt
from webvtt.structures import Caption
import pysrt
from beartype import beartype

if __name__ == "__main__":
    # Assuming config.config contains load_opts and the necessary configuration
    from config.config import *
    opts = load_opts()

def format_srt_time(srt_time):
    # Converts pysrt time to WebVTT-style time string: "00:01:02.345"
    return str(srt_time).replace(',', '.')

def load_subs(path):
    ext = Path(path).suffix.lower()
    if ext == '.srt':
        subs = pysrt.open(path)
        return [
            Caption(
                start=format_srt_time(sub.start),
                end=format_srt_time(sub.end),
                text="\n".join(sub.text.splitlines())
            )
            for sub in subs
        ]
    else:  # .vtt
        return list(webvtt.read(path))

@beartype
def get_labels_start_end_time(
        frame_wise_labels: List[int],
        bg_class: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    """Given a single sequence of frame level labels, find: (i) the start index,
    (ii) the end index and (iii) the label, of each contiguous subsequence of labels.
    """
    labels = []
    starts = []
    ends = []
    if not frame_wise_labels:
        return labels, starts, ends
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


@beartype
def f_score(
        recognized: List[int],
        ground_truth: List[int],
        overlap: float,
        bg_class: List[int],
) -> Tuple[float, float, float]:
    """Compute the f-score of a sequence of predicted sequences against a set of ground
    truth annotations.
    """
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x]
                                               for x in range(len(y_label))])

        # Get the best scoring segment
        if IoU.any():
            idx = np.array(IoU).argmax()
            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        else:
            fp +=1

    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


@beartype
def subs2frames(
        subs: List[webvtt.Caption],
        max_time: float,
        fps: int,
        exclude_subs: List[int],
        background_label: int,
) -> List[int]:
    """Convert subtitles into a single sequence of frame-level labels."""
    total_frames = round(fps * max_time)
    if total_frames == 0:
        return []
    frames = [background_label for _ in range(total_frames)]
    for sub_idx, caption in enumerate(subs):
        if sub_idx in exclude_subs:
            continue

        start_time = min(caption._start, max_time)
        end_time = min(caption._end, max_time)

        start_idx = round(fps * start_time)
        end_idx = round(fps * end_time)
        
        # Ensure indices are within bounds
        start_idx = min(start_idx, total_frames)
        end_idx = min(end_idx, total_frames)
        
        if start_idx < end_idx:
            frames[start_idx:end_idx] = [sub_idx for _ in range(end_idx - start_idx)]
    return frames

def _process_video(pred_path, gt_path, vid_id, shift_start, shift_end, fps, MAX_TIME_PAD_SECS, overlaps, BACKGROUND_LABEL, ext_gt, ext_pred):
    """Process a single video's subtitles and compute metrics."""
    video_correct, video_total_frames, video_total_subs = 0, 0, 0
    video_all_offset_start, video_all_offset_end = [], []
    video_all_offset_start_abs, video_all_offset_end_abs = [], []
    video_tp, video_fp, video_fn = np.zeros(len(overlaps)), np.zeros(len(overlaps)), np.zeros(len(overlaps))
    video_msg = f'No subtitles to evaluate for {vid_id}, skipping'
    
    try:
        pred_subs = load_subs(pred_path)
        gt_subs = load_subs(gt_path)
    except Exception as e:
        warnings.warn(f"Could not load subtitles for {vid_id}. Error: {e}")
        return {
            'correct': 0, 'total_frames': 0, 'total_subs': 0, 'all_offset_start': [],
            'all_offset_end': [], 'all_offset_start_abs': [], 'all_offset_end_abs': [],
            'tp': video_tp, 'fp': video_fp, 'fn': video_fn, 'msg': f"Error loading subs for {vid_id}",
        }

    exceptional_misaligned = False
    if len(gt_subs) != len(pred_subs):
        _gt_subs = gt_subs
        _pred_subs = pred_subs
        gt_subs = [sub for sub in gt_subs if not ("[" in sub.text and "]" in sub.text)]
        pred_subs = [sub for sub in pred_subs if not ("[" in sub.text and "]" in sub.text)]
        if len(gt_subs) != len(pred_subs):
            exceptional_misaligned = True
            gt_subs, pred_subs = _gt_subs, _pred_subs
    else:
        excluded_indices = [i for i, sub in enumerate(gt_subs) if '[' in sub.text and ']' in sub.text]
        gt_subs = [sub for i, sub in enumerate(gt_subs) if i not in excluded_indices]
        pred_subs = [sub for i, sub in enumerate(pred_subs) if i not in excluded_indices]

    excluded_indices = [i for i, sub in enumerate(pred_subs) if '{CSLR_EXCLUDED}' in sub.text]
    gt_subs = [sub for i, sub in enumerate(gt_subs) if i not in excluded_indices]
    pred_subs = [sub for i, sub in enumerate(pred_subs) if i not in excluded_indices]

    for sub in pred_subs:
        sub._start += shift_start
        sub._end += shift_end

    msg = (f"Expected num. preds {len(pred_subs)} to match num. gt {len(gt_subs)} for {pred_path}")
    if exceptional_misaligned: warnings.warn(msg, UserWarning)
    else: assert len(pred_subs) == len(gt_subs), msg

    if len(gt_subs) > 0:
        video_total_subs += len(gt_subs)
        max_time = gt_subs[-1]._end + MAX_TIME_PAD_SECS
        exclude_subs = []
        for sub_idx, sub in enumerate(gt_subs):
            if "[" in sub.text and "]" in sub.text:
                exclude_subs.append(sub_idx)
            else:
                if exceptional_misaligned and sub_idx >= len(pred_subs): continue
                video_all_offset_start.append(sub._start - pred_subs[sub_idx]._start)
                video_all_offset_end.append(sub._end - pred_subs[sub_idx]._end)
                video_all_offset_start_abs.append(abs(sub._start - pred_subs[sub_idx]._start))
                video_all_offset_end_abs.append(abs(sub._end - pred_subs[sub_idx]._end))
        video_total_subs -= len(exclude_subs)

        pred_frames = subs2frames(pred_subs, float(max_time), fps, exclude_subs if exceptional_misaligned else [], BACKGROUND_LABEL)
        gt_frames = subs2frames(gt_subs, float(max_time), fps, exclude_subs, BACKGROUND_LABEL)

        if len(pred_frames) != len(gt_frames):
            warnings.warn(f"Frame sequence length mismatch for {vid_id}: Pred {len(pred_frames)}, GT {len(gt_frames)}. Truncating to shorter length.")
            min_len = min(len(pred_frames), len(gt_frames))
            pred_frames, gt_frames = pred_frames[:min_len], gt_frames[:min_len]

        video_total_frames = len(gt_frames)
        video_correct = sum(1 for pred, gt in zip(pred_frames, gt_frames) if pred == gt)

        for ii, overlap in enumerate(overlaps):
            tp1, fp1, fn1 = f_score(pred_frames, gt_frames, overlap, [BACKGROUND_LABEL])
            video_tp[ii] += tp1
            video_fp[ii] += fp1
            video_fn[ii] += fn1

        if video_total_subs > 0:
            video_msg = (
                f"Mean/median start offset: {mean(video_all_offset_start):.2f}/{median(video_all_offset_start):.2f}\n"
                f"Mean/median end offset: {mean(video_all_offset_end):.2f}/{median(video_all_offset_end):.2f}\n"
                f"Mean/median start offset (abs): {mean(video_all_offset_start_abs):.2f}/{median(video_all_offset_start_abs):.2f}\n"
                f"Mean/median end offset (abs): {mean(video_all_offset_end_abs):.2f}/{median(video_all_offset_end_abs):.2f}\n"
                f"Frames: {video_total_frames}, Sentences: {video_total_subs} - "
                f"Frame Acc: {100 * video_correct / video_total_frames if video_total_frames > 0 else 0:.2f}"
            )
            for ii, overlap in enumerate(overlaps):
                precision = video_tp[ii] / (video_tp[ii] + video_fp[ii]) if (video_tp[ii] + video_fp[ii]) > 0 else 0
                recall = video_tp[ii] / (video_tp[ii] + video_fn[ii]) if (video_tp[ii] + video_fn[ii]) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                video_msg += f" F1@{overlap:0.2f}: {f1 * 100:.2f}"
        
    return {
        'correct': video_correct, 'total_frames': video_total_frames, 'total_subs': video_total_subs,
        'all_offset_start': video_all_offset_start, 'all_offset_end': video_all_offset_end,
        'all_offset_start_abs': video_all_offset_start_abs, 'all_offset_end_abs': video_all_offset_end_abs,
        'tp': video_tp, 'fp': video_fp, 'fn': video_fn, 'msg': video_msg,
    }

def eval_subtitle_alignment(
        pred_path_root: "Path", gt_anno_path_root: "Path", list_videos: list, fps: int,
        shift_start=0, shift_end=0, num_workers=1, debug=False, fps_map: dict = None,
):
    """Evaluate subtitle alignment quality."""
    if not list_videos:
        return "No videos found to evaluate."
    
    if os.path.exists(os.path.join(gt_anno_path_root, list_videos[0]+'.vtt')): ext_gt = '.vtt'
    elif os.path.exists(os.path.join(gt_anno_path_root, list_videos[0]+'.srt')): ext_gt = '.srt'
    else: ext_gt = '/signhd.vtt'

    if os.path.exists(os.path.join(pred_path_root, list_videos[0]+'.vtt')): ext_pred = '.vtt'
    elif os.path.exists(os.path.join(pred_path_root, list_videos[0]+'.srt')): ext_pred = '.srt'
    else: ext_pred = '/signhd.vtt'

    gt_anno_paths = [f'{gt_anno_path_root}/{p}{ext_gt}' for p in list_videos]
    pred_paths = [f'{pred_path_root}/{p}{ext_pred}' for p in list_videos]

    fps_list = [int(fps_map.get(vid_id, fps)) for vid_id in list_videos] if fps_map else [fps] * len(list_videos)

    correct, total, total_subs = 0, 0, 0
    all_offset_start, all_offset_end, all_offset_start_abs, all_offset_end_abs = [], [], [], []
    BACKGROUND_LABEL, MAX_TIME_PAD_SECS = -1, 10
    overlaps = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    results = []
    args_list = [
        (pred_path, gt_path, vid_id, shift_start, shift_end, video_fps,
         MAX_TIME_PAD_SECS, overlaps, BACKGROUND_LABEL, ext_gt, ext_pred)
        for pred_path, gt_path, vid_id, video_fps in zip(pred_paths, gt_anno_paths, list_videos, fps_list)
    ]

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for res in tqdm.tqdm(pool.starmap(_process_video, args_list), total=len(args_list)):
                results.append(res)
    else:
        for args in tqdm.tqdm(args_list):
            res = _process_video(*args)
            results.append(res)
            if debug:
                print(f"Video {args[2]} evaluation (FPS: {args[5]}):\n{res['msg']}\n")

    if debug and num_workers > 1:
        for i, res in enumerate(results):
             print(f"Video {list_videos[i]} evaluation (FPS: {fps_list[i]}):\n{res['msg']}\n")

    for res in results:
        correct += res['correct']
        total += res['total_frames']
        total_subs += res['total_subs']
        all_offset_start.extend(res['all_offset_start'])
        all_offset_end.extend(res['all_offset_end'])
        all_offset_start_abs.extend(res['all_offset_start_abs'])
        all_offset_end_abs.extend(res['all_offset_end_abs'])
        tp += res['tp']
        fp += res['fp']
        fn += res['fn']
    
    if total == 0:
        return "Evaluation complete. No frames were processed."

    msg = (
        f"Mean/median start offset: {mean(all_offset_start):.2f}/{median(all_offset_start):.2f}\n"
        f"Mean/median end offset: {mean(all_offset_end):.2f}/{median(all_offset_end):.2f}\n"
        f"Mean/median start offset (abs): {mean(all_offset_start_abs):.2f}/{median(all_offset_start_abs):.2f}\n"
        f"Mean/median end offset (abs): {mean(all_offset_end_abs):.2f}/{median(all_offset_end_abs):.2f}\n"
        f"Computed over {total} frames, {total_subs} sentences - "
        f"Frame-level accuracy: {100 * correct/total:.2f}"
    )
    for ii, overlap in enumerate(overlaps):
        precision = tp[ii] / (tp[ii] + fp[ii]) if (tp[ii] + fp[ii]) > 0 else 0
        recall = tp[ii] / (tp[ii] + fn[ii]) if (tp[ii] + fn[ii]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        msg += f" F1@{overlap:0.2f}: {f1 * 100:.2f}"

    return msg

def main():
    # All arguments are now sourced from the 'opts' object.
    opts = load_opts()

    # Load per-video FPS if 'fps_file' is specified in opts
    fps_map = {}
    fps_file_path = getattr(opts, 'fps_file', None)  # Safely get the fps_file path
    if fps_file_path:
        try:
            with open(Path(fps_file_path), 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if not row: continue
                    filename, video_fps_str = row
                    video_id = os.path.splitext(filename)[0]
                    fps_map[video_id] = int(float(video_fps_str))
        except FileNotFoundError:
            print(f"Warning: FPS file not found at {fps_file_path}. Using global FPS.")
            fps_map = {}
        except Exception as e:
            print(f"Warning: Error reading FPS file: {e}. Using global FPS.")
            fps_map = {}

    # Read the list of test files robustly
    with open(opts.test_videos_txt, "r") as f:
        test_files = [line.strip() for line in f if line.strip()]

    eval_str = eval_subtitle_alignment(
        pred_path_root=Path(f'{opts.pred_path_root}'),
        gt_anno_path_root=Path(f'{opts.gt_sub_path}'),
        list_videos=test_files,
        fps=opts.fps,
        shift_start=opts.pr_subs_delta_bias_start,
        shift_end=opts.pr_subs_delta_bias_end,
        fps_map=fps_map,
        num_workers=getattr(opts, 'num_workers', 1),
        debug=getattr(opts, 'debug', False),
    )
    print(eval_str)


if __name__ == "__main__":
    main()