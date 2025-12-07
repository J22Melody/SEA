from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Shift subtitle timings, align cues to SIGN segments, merge additional sign annotations (if set), and write an updated ELAN file."
    )

    # ----- Video ID Files -----
    parser.add_argument("--video_ids", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_val_1.txt",
                        help="Path to text file with video ids (one per line).")
    parser.add_argument("--video_ids_train", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_train.txt",
                        help="Path to text file with training video ids.")
    parser.add_argument("--video_ids_val", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_val.txt",
                        help="Path to text file with validation video ids.")
    parser.add_argument("--video_ids_test", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_test.txt",
                        help="Path to text file with test video ids.")

    # ----- Subtitle File Paths -----
    parser.add_argument("--pr_sub_path", type=str,
                        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
                        help="Directory where subtitle (VTT) files are stored.")
    parser.add_argument("--gt_sub_path", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
                        help="Directory where ground truth subtitle (VTT) files are stored.")
    parser.add_argument("--save_dir", type=str,
                        default="/users/zifan/subtitle_align/alternative/aligned_subtitles",
                        help="Directory to store aligned subtitle VTT files.")

    # ----- Subtitle Timing Biases -----
    parser.add_argument("--pr_subs_delta_bias_start", type=float, nargs='+',
                        default=[2.7],
                        help="Delta bias (seconds) added to the start time of each subtitle cue (pre-alignment).")
    parser.add_argument("--pr_subs_delta_bias_end", type=float, nargs='+',
                        default=[2.7],
                        help="Delta bias (seconds) added to the end time of each subtitle cue (pre-alignment).")
    parser.add_argument("--post_subs_delta_bias_start", type=float, nargs='+',
                        default=[0.0],
                        help="Delta bias (seconds) added to the start time of each subtitle cue (post-alignment).")
    parser.add_argument("--post_subs_delta_bias_end", type=float, nargs='+',
                        default=[1.0],
                        help="Delta bias (seconds) added to the end time of each subtitle cue (post-alignment).")

    # ----- Segmentation Files -----
    parser.add_argument("--segmentation_dir", type=str,
                        default="/scratch/shared/beegfs/zifan/bobsl/segmentation",
                        help="Directory with segmentation ELAN (.eaf) files.")
    
    # ----- Segmentation Model and Thresholds -----
    parser.add_argument("--segmentation_model", nargs='+', type=str,
                        default=["model_E4s-1.pth"],
                        help="Path(s) to segmentation model file(s).")
    parser.add_argument("--sign-b-threshold", nargs='+', type=int,
                        default=[30],
                        help="Threshold(s) for sign B.")
    parser.add_argument("--sign-o-threshold", nargs='+', type=int,
                        default=[70],
                        help="Threshold(s) for sign O.")

    # ----- Refinement Options -----
    parser.add_argument("--refine", action="store_true",
                        help="Enable refinement of segmentation.")
    parser.add_argument("--refine_dir", type=str,
                        default="/users/zifan/BOBSL/derivatives/segmentation/E4s-cslr_30_50",
                        help="Directory with segmentation ELAN (.eaf) files for refinement.")

    # ----- CMPL Segmentation Options -----
    parser.add_argument("--cmpl", action="store_true",
                        help="Merge additional sign segments from the CMPL segmentation directory.")
    parser.add_argument("--cmpl_only", action="store_true",
                        help="Use CMPL segmentation only.")
    parser.add_argument("--cmpl_dir", type=str,
                        default="/users/zifan/BOBSL/derivatives/segmentation_bsl/mstcn_bsl1k_cmpl",
                        help="Directory with segmentation files in CMPL format (optional).")
    parser.add_argument("--cmpl_overlapIoU", type=float, nargs='+',
                        default=[0.5],
                        help="IoU threshold for merging CMPL sign segments conservatively.")

    # ----- Pseudo-Glosses Segmentation Options -----
    parser.add_argument("--pseudo_glosses", action="store_true",
                        help="Merge additional sign segments from the pseudo_glosses segmentation directory.")
    parser.add_argument("--pseudo_glosses_only", action="store_true",
                        help="Use pseudo_glosses segmentation only.")
    parser.add_argument("--pseudo_glosses_dir", type=str,
                        default="/users/zifan/BOBSL/derivatives/pseudo_glosses",
                        help="Directory with segmentation files in pseudo_glosses format (optional).")

    # ----- CSLR Options -----
    parser.add_argument("--cslr", action="store_true",
                        help="Merge CSLR CSV sign annotations with ELAN signs before DP alignment.")
    parser.add_argument("--cslr_only", action="store_true",
                        help="Use CSLR segmentation only.")
    parser.add_argument("--cslr_partial_eval", action="store_true",
                        help="Filter subtitle cues to those overlapping with CSLR signs only (partial evaluation).")
    parser.add_argument("--cslr_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/continuous_sign_sequences/cslr-fixed",
                        help="Directory with CSLR CSV files (searched recursively).")

    # ----- Similarity and Embedding Options -----
    parser.add_argument("--similarity_measure", type=str, nargs='+',
                        default=["none"],
                        choices=["none", "cslr_subtitle", "cslr_text", "text_embedding", "sign_clip_embedding"],
                        help="Similarity measure(s) to use. Accepts multiple values.")
    parser.add_argument("--subtitle_embedding_dir", type=str, nargs='+',
                        default=["/scratch/shared/beegfs/zifan/bobsl/subtitle_embedding/sign_clip"],
                        help="Directory/directories containing subtitle cue embeddings (NPY files) for sign_clip_embedding.")
    parser.add_argument("--segmentation_embedding_dir", type=str, nargs='+',
                        default=["/scratch/shared/beegfs/zifan/bobsl/segmentation_embedding/E4s-1_30_50/sign_clip"],
                        help="Directory/directories containing sign segment embeddings (NPY files) for sign_clip_embedding.")
    parser.add_argument("--tokenize_text_embedding", action="store_true",
                        help="Enable tokenization for text embedding.")

    # ----- Mode and Iteration Options -----
    parser.add_argument("--mode", type=str, default="inference",
                        choices=["inference", "training", "dev"],
                        help="Mode: inference (default), training (parameter search), or dev (evaluate on train, val, test).")
    parser.add_argument("--num_search", type=int, default=10,
                        help="Number of random search iterations in training mode.")

    # ----- DP Alignment Options -----
    parser.add_argument("--dp_duration_penalty_weight", type=float, nargs='+',
                        default=[5.0],
                        help="Duration penalty weight(s) for DP alignment.")
    parser.add_argument("--dp_gap_penalty_weight", type=float, nargs='+',
                        default=[10.0],
                        help="Gap penalty weight(s) for DP alignment.")
    parser.add_argument("--dp_window_size", type=int, nargs='+',
                        default=[50],
                        help="Window size(s) for DP alignment.")
    parser.add_argument("--dp_max_gap", type=float, nargs='+',
                        default=[8.0],
                        help="Max gap(s) allowed between SIGN segments for DP alignment.")
    parser.add_argument("--similarity_weight", type=float, nargs='+',
                        default=[30.0],
                        help="Similarity weight(s) for DP alignment.")

    # ----- Processing Options -----
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second of the video.")
    parser.add_argument("--fps_file", type=Path, default=None, help="Path to a CSV file mapping video IDs to their FPS. If provided, overrides the global --fps for specific videos.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of processes for parallel processing.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing files if set.")
    parser.add_argument("--live_segmentation", action="store_true",
                        help="Run live segmentation before alignment.")
    parser.add_argument("--live_embedding", action="store_true",
                        help="Run live embedding before alignment.")
    parser.add_argument("--include_non_sign", action="store_true",
                        help="Include non-sign elements in processing.")

    # ----- Debug Options -----
    parser.add_argument("--visualize_similarity", action="store_true",
                        help="Visualize similarity measures.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, process only the first 20-30 seconds of cues and segments for debugging.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
