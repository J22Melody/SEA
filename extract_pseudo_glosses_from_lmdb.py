#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import csv
import torch
import pickle
import numpy as np
from collections import defaultdict
from operator import itemgetter
from einops import rearrange

# Ensure that the directory containing 'data' is in sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(parent_dir, 'data')
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)

from lmdb_loader import LMDBLoader

def compress_and_average(labels, probs):
    """
    Compress duplicate labels and average their probabilities.
    labels: list of strings
    probs: list of floats
    Returns a tuple (final_labels, final_probs) with unique labels in order of appearance.
    """
    groups = defaultdict(list)
    for label, prob in zip(labels, probs):
        groups[label].append(prob)
    avg = {label: sum(ps) / len(ps) for label, ps in groups.items()}
    seen = set()
    final_labels = []
    final_probs = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            final_labels.append(label)
            final_probs.append(avg[label])
    return final_labels, final_probs

def fix_synonyms_dict(syn_dict):
    """
    Ensure that the synonyms dictionary satisfies:
      - if a is a synonym of b, then b is a synonym of a
      - a is a synonym of a
    """
    for word, syns in syn_dict.items():
        if word not in syns:
            syns.append(word)
        for syn in syns:
            if word not in syn_dict[syn]:
                syn_dict[syn].append(word)
    return syn_dict

def synonym_combine(candidates, probs_tensor):
    """
    Aggregate probabilities of synonyms.
    candidates: np.ndarray of candidate strings (e.g. shape (k,))
    probs_tensor: torch.Tensor of shape (k,) containing probabilities.
    
    Returns (new_probs, new_candidates) where:
      - new_probs is a torch.Tensor of aggregated probabilities.
      - new_candidates is a numpy array of candidate strings.
    """
    change = False
    new_probs_list = []
    for anchor_idx, anchor in enumerate(candidates):
        try:
            anchor_clean = anchor.replace("-", " ")
            syns = synonyms_dict[anchor_clean]
            anchor_new_prob = 0
            for checked_idx, checked_label in enumerate(candidates):
                checked_label_clean = checked_label.replace("-", " ")
                if checked_label_clean in syns:
                    anchor_new_prob += probs_tensor[checked_idx]
                if checked_idx != anchor_idx:
                    change = True
            new_probs_list.append(anchor_new_prob)
        except KeyError:
            new_probs_list.append(probs_tensor[anchor_idx])
    if change:
        sorted_indices = torch.argsort(-1 * torch.tensor(new_probs_list))
        new_probs_tensor = torch.tensor(new_probs_list)[sorted_indices]
        new_candidates = np.array(candidates)[sorted_indices]
    else:
        new_probs_tensor = torch.tensor(new_probs_list)
        new_candidates = np.array(candidates)
    return new_probs_tensor, new_candidates

# Global variable for synonyms dictionary (loaded from synonyms_pkl)
synonyms_dict = None

def main():
    parser = argparse.ArgumentParser(
        description="Extract pseudo glosses from LMDB for a list of videos with synonym grouping enabled."
    )
    parser.add_argument("--video_ids", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_test.txt",
                        help="Path to text file with video ids (one per line).")
    parser.add_argument("--video_path", type=str,
                        default="/users/zifan/BOBSL/derivatives/original_videos",
                        help="Path to directory containing video files.")
    parser.add_argument("--vocab_path", type=str,
                        default="/work/sign-language/haran/bobsl/vocab/8697_vocab.pkl",
                        help="Path to vocabulary file.")
    parser.add_argument("--lmdb_path", type=str,
                        default="/users/zifan/BOBSL/v1.4/automatic_annotations/continuous_sign_sequences/swin_v2_pseudo_labels/lmdb-pl_vswin_t-bs256_float16",
                        help="Path to LMDB file for pseudo glosses.")
    parser.add_argument("--output_dir", type=str,
                        default="/users/zifan/BOBSL/derivatives/pseudo_glosses/",
                        help="Output directory for the generated CSV files.")
    parser.add_argument("--pl_filter", type=float, default=0.6,
                        help="Probability filter threshold for pseudo glosses.")
    parser.add_argument("--pl_min_count", type=int, default=6,
                        help="Minimum count threshold for pseudo glosses.")
    # Synonym grouping turned on by default.
    parser.add_argument("--synonym_grouping", action="store_true",
                        help="Enable synonym grouping (default: enabled).")
    parser.add_argument("--synonyms_pkl", type=str, default='/users/zifan/work/sign-language/data_from_craude/bobsl/pickles/synonyms_t5_large_.8.pkl',
                        help="Path to synonyms pickle file.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode to print debug info and exit.")
    
    args = parser.parse_args()

    # Create output directory if needed.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vocabulary and build inverted_vocab mapping.
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    if "words_to_id" in vocab:
        vocab = vocab["words_to_id"]
    inverted_vocab = {v: k for k, v in vocab.items()}

    # Read video IDs from the text file.
    with open(args.video_ids, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]

    # Load synonyms dictionary if synonym grouping is enabled.
    global synonyms_dict
    if args.synonym_grouping:
        with open(args.synonyms_pkl, "rb") as f:
            synonyms_dict = pickle.load(f)
        synonyms_dict = fix_synonyms_dict(synonyms_dict)

    # Initialize LMDBLoader once using parameters similar to sentence.py.
    pseudo_label_loader = LMDBLoader(
        lmdb_path=args.lmdb_path,
        load_stride=1,
        load_float16=False,
        load_type="pseudo-labels",
        verbose=False,
        lmdb_window_size=16,
        lmdb_stride=2
    )

    for video_id in video_ids:
        print(f"Processing video: {video_id}")
        video_file = os.path.join(args.video_path, f"{video_id}.mp4")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}. Skipping...")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Video {video_id}: {frame_count} frames at {fps:.2f} FPS")

        # For debug mode, use end_frame=10000; otherwise use full frame_count.
        start_sec = 0.0
        end_frame = 10000 if args.debug else frame_count

        # Load pseudo gloss sequence.
        labels, probs = pseudo_label_loader.load_sequence(
            episode_name=video_id,
            begin_frame=int(start_sec * fps),
            end_frame=end_frame
        )
        if labels is None or probs is None or len(labels) == 0:
            print(f"No pseudo glosses found for video {video_id}.")
            continue

        # Apply synonym grouping if enabled.
        if args.synonym_grouping:
            # Convert labels to words using inverted_vocab.
            words = itemgetter(*rearrange(labels.numpy(), "t k -> (t k)"))(inverted_vocab)
            words = rearrange(np.array(words), "(t k) -> t k", k=5)
            new_words, new_probs = [], []
            for word, prob in zip(words, probs):
                new_prob, new_word = synonym_combine(word, prob)
                new_words.append(new_word)
                new_probs.append(new_prob)
            # new_words is already an array of shape (T, k); no need to rearrange.
            new_words = np.array(new_words)
            # Convert new words back to label ids using vocab.
            labels = itemgetter(*new_words)(vocab)
            labels = rearrange(torch.tensor(labels), "(t k) -> t k", k=5)
            probs = torch.stack(new_probs)
            if args.debug:
                print("After synonym grouping, first 5 labels:", labels[:5])

        # New grouping: group consecutive annotations (by annotation index) that share the same word.
        annotations_list = []
        for i, (prob_tensor, label_tensor) in enumerate(zip(probs, labels)):
            prob_val = prob_tensor[0].item() if isinstance(prob_tensor, torch.Tensor) else float(prob_tensor)
            if prob_val >= args.pl_filter:
                # Calculate the actual frame number using lmdb_stride.
                frame_number = int(start_sec * fps) + i * pseudo_label_loader.lmdb_stride
                label_val = label_tensor[0].item() if isinstance(label_tensor, torch.Tensor) else label_tensor
                word = inverted_vocab.get(label_val, str(label_val))
                annotations_list.append((i, frame_number, word, prob_val))
        if args.debug:
            print("Total annotations after filtering by pl_filter:", len(annotations_list))
        
        # Group consecutive annotations that have the same word.
        groups = []
        if annotations_list:
            current_group = [annotations_list[0]]
            for ann in annotations_list[1:]:
                # Group if the word is the same and the annotation index is consecutive.
                if ann[2] == current_group[-1][2] and ann[0] == current_group[-1][0] + 1:
                    current_group.append(ann)
                else:
                    groups.append(current_group)
                    current_group = [ann]
            groups.append(current_group)
        
        # Build pseudo glosses from groups.
        pseudo_glosses = []
        for group in groups:
            first = group[0]
            last = group[-1]
            # Gloss start is the time of the first annotation.
            gloss_start = first[1] / fps
            # Gloss end is the time of the last annotation plus one stride.
            gloss_end = (last[1] + pseudo_label_loader.lmdb_stride) / fps
            avg_prob = sum(x[3] for x in group) / len(group)
            pseudo_glosses.append({
                "start": gloss_start,
                "end": gloss_end,
                "text": first[2],
                "probs": avg_prob
            })

        if args.debug:
            print("DEBUG: Final pseudo glosses:")
            for gloss in pseudo_glosses:
                print(gloss)
            sys.exit(0)
        else:
            output_csv = os.path.join(args.output_dir, f"{video_id}.csv")
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["start", "end", "text", "probs"])
                writer.writeheader()
                for row in pseudo_glosses:
                    writer.writerow(row)
            print(f"Written pseudo glosses for video {video_id} to {output_csv}")

if __name__ == "__main__":
    main()
