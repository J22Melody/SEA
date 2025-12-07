#!/usr/bin/env python3
import os
import sys
import random
import argparse
import numpy as np
import re
import csv
import glob
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import subprocess
from itertools import product
import xml.etree.ElementTree as ET

from utils import (
    shift_cues,
    get_subtitle_cues,
    reconstruct_vtt,
    get_sign_segments_from_eaf,
    write_updated_eaf,
    print_results,
    extract_f1_score,
    get_cslr_signs,
    get_cmpl_signs,
    get_pseudo_signs,
    merge_signs,
    filter_cues_by_cslr,
)
from config import get_args  # Import argument parser from config.py

# Add the ../misc directory to sys.path to import the evaluation function.
current_dir = os.path.dirname(os.path.abspath(__file__))
misc_dir = os.path.join(current_dir, "./misc")
if misc_dir not in sys.path:
    sys.path.append(misc_dir)
from evaluate_sub_alignment import eval_subtitle_alignment

from align_similarity import compute_similarity_matrix
from align_dp import dp_align_subtitles_to_signs

def process_video(video_id, args, dp_duration_penalty_weight, dp_gap_penalty_weight,
                  dp_window_size, dp_max_gap, similarity_weight, output_dir, save_elan,
                  seg_model, seg_sign_b, seg_sign_o,
                  pr_subs_delta_bias_start, pr_subs_delta_bias_end,
                  post_subs_delta_bias_start, post_subs_delta_bias_end, cmpl_overlapIoU):
    print(f"Processing video: {video_id}")
    # Derive a cleaned segmentation model name: remove "model_" prefix and ".pth" suffix.
    seg_model_name = seg_model
    if seg_model_name.startswith("model_"):
        seg_model_name = seg_model_name[len("model_"):]
    if seg_model_name.endswith(".pth"):
        seg_model_name = seg_model_name[:-4]
    
    # Always set segmentation_file (using the default segmentation_dir) so that write_updated_eaf works.
    seg_subdir = os.path.join(args.segmentation_dir, f"{seg_model_name}_{seg_sign_b}_{seg_sign_o}")
    segmentation_file = os.path.join(seg_subdir, f"{video_id}.eaf")
    
    if os.path.exists(segmentation_file):
        elan_signs = get_sign_segments_from_eaf(segmentation_file)
    else:
        raise FileNotFoundError(f"Segmentation {segmentation_file} does not exist!")
    
    # If --cmpl is set, merge in additional signs from the CMPL segmentation directory,
    # using the passed overlapIoU threshold (cmpl_overlapIoU).
    cmpl_signs = get_cmpl_signs(video_id, args.cmpl_dir)
    if args.cmpl:
        if cmpl_signs:
            if args.cmpl_only:
                elan_signs = cmpl_signs
            else:
                elan_signs = merge_signs(elan_signs, cmpl_signs, conservative=True, overlapIoU=cmpl_overlapIoU)

    refine_signs = None
    if args.refine:
        refine_signs = get_sign_segments_from_eaf(os.path.join(args.refine_dir, f"{video_id}.eaf"))
        elan_signs = merge_signs(elan_signs, refine_signs, conservative=True)

    pseudo_signs = None
    if args.pseudo_glosses:
        pseudo_signs = get_pseudo_signs(video_id, args.pseudo_glosses_dir)
        if args.pseudo_glosses_only:
            elan_signs = pseudo_signs
        else:
            elan_signs = merge_signs(elan_signs, pseudo_signs, conservative=True)
    
    # If --cslr is set, merge in additional signs from the CSLR CSV files.
    cslr_signs = get_cslr_signs(video_id, args.cslr_dir)
    if args.cslr and cslr_signs:
        if args.cslr_only:
            elan_signs = cslr_signs
        else:
            elan_signs = merge_signs(elan_signs, cslr_signs, conservative=False)
    
    signs = elan_signs

    # Find and load the predicted subtitle file (.vtt or .srt)
    subtitle_file = None
    for ext in ['.vtt', '.srt']:
        candidate = os.path.join(args.pr_sub_path, f"{video_id}{ext}")
        if os.path.exists(candidate):
            subtitle_file = candidate
            break

    if not subtitle_file:
        print(f"Subtitle file for video {video_id} not found. Skipping.")
        return

    header_lines, cues = get_subtitle_cues(subtitle_file)
    if not cues:
        return

    # Find and load the ground truth subtitle file (.vtt or .srt)
    gt_subtitle_file = None
    gt_cues = None  # changed: None denotes "not available"
    for ext in ['.vtt', '.srt']:
        candidate = os.path.join(args.gt_sub_path, f"{video_id}{ext}")
        if os.path.exists(candidate):
            gt_subtitle_file = candidate
            break

    if gt_subtitle_file:
        try:
            _, gt_cues = get_subtitle_cues(gt_subtitle_file)
        except Exception as e:
            print(f"Failed to read GT subtitle for {video_id}: {e}")
            gt_cues = None  # changed: ensure None if parsing fails

    # Prepare for optional non-sign filtering
    excluded_ids = []  # changed: define upfront so later usage is safe

    if not args.include_non_sign:
        if gt_cues:
            # Ensure both lists have the same length before filtering; otherwise skip filtering.
            if len(cues) != len(gt_cues):
                print(f"Warning: len(cues) {len(cues)} != len(gt_cues) {len(gt_cues)} for {video_id}. Skipping non-sign filtering.")
            else:
                filtered_cues = []
                for i, cue in enumerate(cues):
                    if '[' not in gt_cues[i]['text'] and ']' not in gt_cues[i]['text']:
                        filtered_cues.append(cue)
                    else:
                        excluded_ids.append(i)
                cues = filtered_cues
        else:
            # No GT present: skip filtering
            print(f"No GT subtitles for {video_id}. Skipping non-sign filtering.")

    # Apply pre-alignment bias on cues.
    cues = shift_cues(cues, pr_subs_delta_bias_start, pr_subs_delta_bias_end)

    # Initialize the output similarity matrix.
    # If the only similarity measure is "none", then sim_matrices will be set to None.
    if args.similarity_measure == ["none"]:
        sim_matrix = None
    else:
        sim_matrices = []  # List to store similarity matrices for each measure.
        
        # Loop over each provided similarity measure.
        for i, sim_measure in enumerate(args.similarity_measure):
            # For the "sign_clip_embedding" measure, we need to get the embeddings first.
            if sim_measure == "sign_clip_embedding":
                if args.live_embedding:
                    # If live embedding is enabled, import and run the live embedding functions.
                    sys.path.append("/users/zifan/sign_clip/scripts_bsl")
                    from extract_episode_features import live_embed_subtitles, live_embed_signs

                    subtitle_embedding, subtitle_embedding_tokenized = live_embed_subtitles(
                        cues, tokenize_text_embedding=args.tokenize_text_embedding
                    )
                    segmentation_embedding = live_embed_signs(signs, video_id)
                else:
                    # Use the corresponding directory by index.
                    subdir = args.subtitle_embedding_dir[i] 
                    segdir = args.segmentation_embedding_dir[i] 
                    subtitle_emb_file = os.path.join(subdir, f"{video_id}.npy")
                    segmentation_emb_file = os.path.join(segdir, f"{video_id}.npy")
                    
                    if os.path.exists(subtitle_emb_file) and os.path.exists(segmentation_emb_file):
                        subtitle_embedding = np.load(subtitle_emb_file)
                        # Optionally remove non-sign embeddings if specified.
                        if (not args.include_non_sign) and excluded_ids:  # changed: guard on excluded_ids
                            subtitle_embedding = np.delete(subtitle_embedding, excluded_ids, axis=0)
                        segmentation_embedding = np.load(segmentation_emb_file)
                    else:
                        print(f"Embedding files for video {video_id} not found for similarity measure '{sim_measure}' in directories: {subdir} and {segdir}. Skipping measure.")
                        continue  # Skip this measure if files are not found.
                    
                    # Since we are not live embedding, set tokenized embeddings to None.
                    subtitle_embedding_tokenized = None
                
                # Call compute_similarity_matrix with the embeddings.
                sim_matrix = compute_similarity_matrix(
                    cues, signs, sim_measure,
                    subtitle_embedding, subtitle_embedding_tokenized, segmentation_embedding,
                    tokenize_text_embedding=args.tokenize_text_embedding
                )
            else:
                # For non-sign_clip_embedding measures, compute similarity directly.
                sim_matrix = compute_similarity_matrix(
                    cues, signs, sim_measure,
                    tokenize_text_embedding=args.tokenize_text_embedding
                )
            sim_matrices.append(sim_matrix)

        # Convert list to numpy array and average along the 0th axis (elementwise mean).
        sim_matrix = np.mean(np.array(sim_matrices), axis=0)

    # Debug slicing with safe GT handling
    gt_list = gt_cues if gt_cues else []  # changed: normalize to list for downstream
    if args.debug:
        debug_sec = 30
        cues_ = [cue for cue in cues if cue['start'] < debug_sec]
        gt_cues_ = [cue for cue in gt_list if cue['start'] < debug_sec]
        signs_ = [seg for seg in signs if seg['start'] < debug_sec]
    else:
        cues_, gt_cues_, signs_ = cues, gt_list, signs  # changed: use gt_list

    dp_align_subtitles_to_signs(cues_, signs_, gt_cues=gt_cues_,
       duration_penalty_weight=dp_duration_penalty_weight,
       gap_penalty_weight=dp_gap_penalty_weight,
       window_size=dp_window_size,
       max_gap=dp_max_gap,
       similarity_weight=similarity_weight,
       sim_matrix=sim_matrix,
       visualize_similarity=args.visualize_similarity)
    
    # Apply post-alignment bias on the cues.
    cues = shift_cues(cues, post_subs_delta_bias_start, post_subs_delta_bias_end)
    
    # If --cslr_partial_eval is set, filter cues to only those overlapping with CSLR signs.
    if args.cslr_partial_eval:
        cues = filter_cues_by_cslr(cues, cslr_signs)
    
    updated_vtt = reconstruct_vtt(header_lines, cues)
    output_vtt = os.path.join(output_dir, f"{video_id}.vtt")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_vtt, "w", encoding="utf-8") as fout:
            fout.write(updated_vtt)
    except Exception:
        pass

    if save_elan and os.path.exists(segmentation_file):
        additional_signs = {'CMPL': cmpl_signs, 'REFINE': refine_signs, 'CSLR': cslr_signs}
        write_updated_eaf(segmentation_file, cues, video_id, signs, additional_signs=additional_signs)

def process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, similarity_weight, output_dir, save_elan,
                       seg_model, seg_sign_b, seg_sign_o,
                       pr_subs_start, pr_subs_end,
                       post_subs_start, post_subs_end, cmpl_overlapIoU):
    # If live_segmentation is set, run segmentation.py as a subprocess.
    if args.live_segmentation:
        seg_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation.py")
        cmd = f"python {seg_script} --video_ids {args.video_ids} --save_dir {args.segmentation_dir} --num_workers {args.num_workers} --model {seg_model} --sign-b-threshold {seg_sign_b} --sign-o-threshold {seg_sign_o}"
        if args.overwrite:
            cmd += " --overwrite"
        print("Running live segmentation subprocess...")
        print(cmd)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print("Error in live segmentation subprocess, aborting alignment.")
            return

    func = partial(process_video, args=args,
                   dp_duration_penalty_weight=dp_dpw,
                   dp_gap_penalty_weight=dp_gpw,
                   dp_window_size=dp_ws,
                   dp_max_gap=dp_mg,
                   similarity_weight=similarity_weight,
                   output_dir=output_dir,
                   save_elan=save_elan,
                   seg_model=seg_model,
                   seg_sign_b=seg_sign_b,
                   seg_sign_o=seg_sign_o,
                   pr_subs_delta_bias_start=pr_subs_start,
                   pr_subs_delta_bias_end=pr_subs_end,
                   post_subs_delta_bias_start=post_subs_start,
                   post_subs_delta_bias_end=post_subs_end,
                   cmpl_overlapIoU=cmpl_overlapIoU)
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(func, video_ids), total=len(video_ids), desc="Processing videos"):
                pass
    else:
        for vid in tqdm(video_ids, desc="Processing videos"):
            func(vid)

def load_video_ids(args, mode):
    """
    Loads video IDs based on the given mode.
    For "inference", reads from args.video_ids.
    For "dev" and "training", reads from args.video_ids_train, args.video_ids_val, and args.video_ids_test.
    Returns a dictionary with keys:
      - "all": combined list (for processing)
      - For dev/training, also returns "train", "val", "test" lists.
    """
    if mode == "inference":
        print('inference')
        with open(args.video_ids, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        return {"all": ids}
    elif mode in ["dev", "training"]:
        with open(args.video_ids_train, "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        with open(args.video_ids_val, "r") as f:
            val_ids = [line.strip() for line in f if line.strip()]
        with open(args.video_ids_test, "r") as f:
            test_ids = [line.strip() for line in f if line.strip()]
        combined = train_ids + val_ids + test_ids
        return {"all": combined, "train": train_ids, "val": val_ids, "test": test_ids}
    else:
        return {"all": []}

def get_alignment_params(args, randomize=False):
    """
    Returns a tuple of common alignment parameters.
    If randomize is True, a random value is selected from each list.
    Otherwise, the first value is used.
    """
    if randomize:
        dp_dpw = random.choice(args.dp_duration_penalty_weight)
        dp_gpw = random.choice(args.dp_gap_penalty_weight)
        dp_ws  = random.choice(args.dp_window_size)
        dp_mg  = random.choice(args.dp_max_gap)
        sim_w  = random.choice(args.similarity_weight)
        seg_model = random.choice(args.segmentation_model)
        seg_sign_b = random.choice(args.sign_b_threshold)
        seg_sign_o = random.choice(args.sign_o_threshold)
        pr_subs_start = random.choice(args.pr_subs_delta_bias_start)
        pr_subs_end   = random.choice(args.pr_subs_delta_bias_end)
        post_subs_start = random.choice(args.post_subs_delta_bias_start)
        post_subs_end   = random.choice(args.post_subs_delta_bias_end)
    else:
        dp_dpw = args.dp_duration_penalty_weight[0]
        dp_gpw = args.dp_gap_penalty_weight[0]
        dp_ws  = args.dp_window_size[0]
        dp_mg  = args.dp_max_gap[0]
        sim_w  = args.similarity_weight[0]
        seg_model = args.segmentation_model[0]
        seg_sign_b = args.sign_b_threshold[0]
        seg_sign_o = args.sign_o_threshold[0]
        pr_subs_start = args.pr_subs_delta_bias_start[0]
        pr_subs_end   = args.pr_subs_delta_bias_end[0]
        post_subs_start = args.post_subs_delta_bias_start[0]
        post_subs_end   = args.post_subs_delta_bias_end[0]
    return (dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, seg_model, seg_sign_b, seg_sign_o,
            pr_subs_start, pr_subs_end, post_subs_start, post_subs_end)

def main():
    args = get_args()  # Load arguments from config.py

    # --- NEW: Load per-video FPS from fps_file if provided ---
    fps_map = {}
    fps_file_path = getattr(args, 'fps_file', None)
    if fps_file_path:
        print(f"Loading per-video FPS from: {fps_file_path}")
        try:
            with open(fps_file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if not row: continue
                    filename, video_fps_str = row
                    # Strip extension from filename to get the video ID
                    video_id = os.path.splitext(filename)[0]
                    fps_map[video_id] = int(float(video_fps_str))
            print(f"Loaded FPS for {len(fps_map)} videos.")
        except FileNotFoundError:
            print(f"Warning: FPS file not found at {fps_file_path}. Using global FPS.")
            fps_map = {}
        except Exception as e:
            print(f"Warning: Error reading FPS file: {e}. Using global FPS.")
            fps_map = {}
    # --- END NEW ---

    mode = args.mode
    # Load video IDs.
    vids_dict = load_video_ids(args, mode)
    # Get alignment parameters.
    (dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, seg_model, seg_sign_b, seg_sign_o,
     pr_subs_start, pr_subs_end, post_subs_start, post_subs_end) = get_alignment_params(args)

    if mode == "inference":
        video_ids = vids_dict["all"]
        process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, args.save_dir, save_elan=True,
                           seg_model=seg_model, seg_sign_b=seg_sign_b, seg_sign_o=seg_sign_o,
                           pr_subs_start=pr_subs_start, pr_subs_end=pr_subs_end,
                           post_subs_start=post_subs_start, post_subs_end=post_subs_end,
                           cmpl_overlapIoU=args.cmpl_overlapIoU[0])

        # changed: Skip eval when any GT subtitle is missing; still save outputs/ELAN.
        missing_gt = [
            vid for vid in video_ids
            if not any(os.path.exists(os.path.join(args.gt_sub_path, f"{vid}{ext}")) for ext in ['.vtt', '.srt'])
        ]
        if missing_gt:
            print(f"Skipping evaluation: missing GT subtitles for {len(missing_gt)} video(s).")
        else:
            eval_output = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                                  video_ids, args.fps, 0, 0, num_workers=args.num_workers,
                                                  fps_map=fps_map)
            print_results(eval_output)
    elif mode == "dev":
        video_ids = vids_dict["all"]
        process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, args.save_dir, save_elan=True,
                           seg_model=seg_model, seg_sign_b=seg_sign_b, seg_sign_o=seg_sign_o,
                           pr_subs_start=pr_subs_start, pr_subs_end=pr_subs_end,
                           post_subs_start=post_subs_start, post_subs_end=post_subs_end,
                           cmpl_overlapIoU=args.cmpl_overlapIoU[0])
        eval_train = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                             vids_dict["train"], args.fps, 0, 0, num_workers=args.num_workers,
                                             fps_map=fps_map) # MODIFIED
        eval_val = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                           vids_dict["val"], args.fps, 0, 0, num_workers=args.num_workers,
                                           fps_map=fps_map) # MODIFIED
        eval_test = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                            vids_dict["test"], args.fps, 0, 0, num_workers=args.num_workers,
                                            fps_map=fps_map) # MODIFIED
        col_names = [os.path.splitext(os.path.basename(p))[0] for p in 
                     [args.video_ids_train, args.video_ids_val, args.video_ids_test]]
        print_results([eval_train, eval_val, eval_test], column_names=col_names)
    elif mode == "training":
        # Use train IDs for parameter search and then final evaluation with all IDs.
        train_ids = vids_dict["train"]
        all_ids = vids_dict["all"]
        training_base = f"{args.save_dir}_training"
        os.makedirs(training_base, exist_ok=True)
        best_score = -1.0
        best_params = None
        scores = {}
        for i in range(args.num_search):
            # For each trial, randomize alignment parameters.
            params = get_alignment_params(args, randomize=True)
            dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, seg_model, seg_sign_b, seg_sign_o, pr_subs_start, pr_subs_end, post_subs_start, post_subs_end = params
            cmpl_overlapIoU = random.choice(args.cmpl_overlapIoU)
            comb_str = (f"dpd_{dp_dpw}_dpg_{dp_gpw}_ws_{dp_ws}_mg_{dp_mg}_sim_{sim_w}_"
                        f"{seg_model}_{seg_sign_b}_{seg_sign_o}_{pr_subs_start}_{pr_subs_end}_"
                        f"{post_subs_start}_{post_subs_end}_cmplIoU_{cmpl_overlapIoU}")
            output_dir = os.path.join(training_base, comb_str)
            os.makedirs(output_dir, exist_ok=True)
            process_all_videos(train_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, output_dir, save_elan=False,
                               seg_model=seg_model, seg_sign_b=seg_sign_b, seg_sign_o=seg_sign_o,
                               pr_subs_start=pr_subs_start, pr_subs_end=pr_subs_end,
                               post_subs_start=post_subs_start, post_subs_end=post_subs_end,
                               cmpl_overlapIoU=cmpl_overlapIoU)
            eval_output = eval_subtitle_alignment(Path(output_dir), Path(args.gt_sub_path),
                                                  train_ids, args.fps, 0, 0, num_workers=args.num_workers,
                                                  fps_map=fps_map) # MODIFIED
            f1_score = extract_f1_score(eval_output)
            scores[comb_str] = f1_score
            print(f"Trial {i+1}/{args.num_search}, Params: {comb_str}, F1@0.50: {f1_score}")
            if f1_score > best_score:
                best_score = f1_score
                best_params = (dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, seg_model, seg_sign_b, seg_sign_o,
                               pr_subs_start, pr_subs_end, post_subs_start, post_subs_end, cmpl_overlapIoU)
                print("New best found!")
                print(f"New Best F1@0.50: {best_score} with parameters: dp_duration_penalty_weight={best_params[0]}, "
                      f"dp_gap_penalty_weight={best_params[1]}, dp_window_size={best_params[2]}, "
                      f"dp_max_gap={best_params[3]}, similarity_weight={best_params[4]}, "
                      f"segmentation_model={best_params[5]}, sign-b-threshold={best_params[6]}, sign-o-threshold={best_params[7]}, "
                      f"pr_subs_delta_bias_start={best_params[8]}, pr_subs_delta_bias_end={best_params[9]}, "
                      f"post_subs_delta_bias_start={best_params[10]}, post_subs_delta_bias_end={best_params[11]}, "
                      f"cmpl_overlapIoU={best_params[12]}")
        print("----- All Trials -----")
        for comb_str, score in scores.items():
            print(f"{comb_str} => F1@0.50: {score}")
        print("----- Best Parameters -----")
        print(f"Best F1@0.50: {best_score} with parameters: dp_duration_penalty_weight={best_params[0]}, "
              f"dp_gap_penalty_weight={best_params[1]}, dp_window_size={best_params[2]}, "
              f"dp_max_gap={best_params[3]}, similarity_weight={best_params[4]}, "
              f"segmentation_model={best_params[5]}, sign-b-threshold={best_params[6]}, sign-o-threshold={best_params[7]}, "
              f"pr_subs_delta_bias_start={best_params[8]}, pr_subs_delta_bias_end={best_params[9]}, "
              f"post_subs_delta_bias_start={best_params[10]}, post_subs_delta_bias_end={best_params[11]}, "
              f"cmpl_overlapIoU={best_params[12]}")
        process_all_videos(all_ids, args, best_params[0], best_params[1], best_params[2],
                           best_params[3], best_params[4], args.save_dir, save_elan=True,
                           seg_model=best_params[5], seg_sign_b=best_params[6], seg_sign_o=best_params[7],
                           pr_subs_start=best_params[8], pr_subs_end=best_params[9],
                           post_subs_start=best_params[10], post_subs_end=best_params[11],
                           cmpl_overlapIoU=best_params[12])
        eval_train = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                             vids_dict["train"], args.fps, 0, 0, num_workers=args.num_workers,
                                             fps_map=fps_map) # MODIFIED
        eval_val = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                           vids_dict["val"], args.fps, 0, 0, num_workers=args.num_workers,
                                           fps_map=fps_map) # MODIFIED
        eval_test = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path),
                                            vids_dict["test"], args.fps, 0, 0, num_workers=args.num_workers,
                                            fps_map=fps_map) # MODIFIED
        col_names = [os.path.splitext(os.path.basename(p))[0] for p in 
                     [args.video_ids_train, args.video_ids_val, args.video_ids_test]]
        print_results([eval_train, eval_val, eval_test], column_names=col_names)
    
if __name__ == '__main__':
    main()
