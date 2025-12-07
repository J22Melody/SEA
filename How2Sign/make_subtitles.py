#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate .vtt subtitle files for How2Sign videos.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/common/How2Sign/sentence_level/val/text/en/raw_text/how2sign_val.csv",
        help="Path to the input How2Sign CSV file containing video IDs and text."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/common/How2Sign/video_level/",
        help="Base directory to save the generated .vtt subtitle files."
    )
    parser.add_argument(
        "--video_id_file",
        type=str,
        default="./data/how2_align_val.txt",
        help="Path to save the list of VIDEO_NAMEs (one per line)."
    )
    return parser.parse_args()


def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    print(f"Reading CSV from: {csv_path}")
    return pd.read_csv(csv_path, sep="\t")


def format_timestamp(seconds):
    """Convert seconds (float) to WebVTT time format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def generate_vtt(df_group, start_col, end_col):
    """Generate VTT subtitle content from grouped dataframe."""
    lines = ["WEBVTT\n"]
    for idx, row in df_group.iterrows():
        start = format_timestamp(row[start_col])
        end = format_timestamp(row[end_col])
        text = row["SENTENCE"]
        lines.append(f"{start} --> {end}")
        lines.append(text.strip())
        lines.append("")  # Blank line after each cue
    return "\n".join(lines)


def save_vtt(content, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {output_path}")


def write_video_ids(video_names, video_id_file):
    output_path = Path(video_id_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(sorted(video_names)):
            if i < len(video_names) - 1:
                f.write(name + "\n")
            else:
                f.write(name)
    print(f"Saved VIDEO_NAME list to: {video_id_file}")


def main():
    args = parse_arguments()

    is_realigned = "re_aligned" in args.csv_path.lower()
    start_col = "START_REALIGNED" if is_realigned else "START"
    end_col = "END_REALIGNED" if is_realigned else "END"

    # Adjust target directory based on input
    sub_dir = "subtitles_manual" if is_realigned else "subtitles_audio"
    final_target_dir = Path(args.target_dir) / sub_dir

    df = load_csv(args.csv_path)

    print(f"Using columns: {start_col}, {end_col}")
    print(f"Total rows loaded: {len(df)}")
    print(f"Output directory: {final_target_dir}")

    grouped = df.sort_values(["VIDEO_NAME", start_col]).groupby("VIDEO_NAME")
    total_files = len(grouped)

    print(f"Generating {total_files} subtitle files...")

    video_names = []

    for i, (video_name, group) in enumerate(grouped, 1):
        print(f"[{i}/{total_files}] Processing VIDEO_NAME: {video_name} ({len(group)} segments)")
        vtt_content = generate_vtt(group, start_col, end_col)
        output_path = final_target_dir / f"{video_name}.vtt"
        save_vtt(vtt_content, output_path)
        video_names.append(video_name)

    write_video_ids(video_names, args.video_id_file)

    print("All VTT files and VIDEO_NAME list have been generated successfully.")


if __name__ == "__main__":
    main()
