#!/usr/bin/env python3

import argparse
from pathlib import Path
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Make parallel SRT files with same subtitle count, filling missing units with [NOT-SIGNED] from segmented."
    )
    parser.add_argument(
        "--segmented_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_segmented",
        help="Path to segmented subtitles (reference)."
    )
    parser.add_argument(
        "--corrected_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_corrected",
        help="Path to corrected subtitles (to be made parallel)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_corrected_parallel",
        help="Output directory for parallel subtitles."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: Only print actions, do not write files."
    )
    return parser.parse_args()

def parse_srt(file_path):
    """Parse SRT file into list of (number, start, end, text)."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip(), flags=re.MULTILINE)
    units = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            num = lines[0]
            times = lines[1]
            m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", times)
            if m:
                start, end = m.groups()
                text = "\n".join(lines[2:]).strip()
                units.append((num, start, end, text))
    return units

def write_srt(units, file_path, debug=False):
    srt_lines = []
    for idx, (num, start, end, text) in enumerate(units, start=1):
        srt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    content = "\n".join(srt_lines)
    if debug:
        print(f"[DEBUG] Would write SRT to {file_path} with {len(units)} units.")
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def main():
    args = parse_args()

    segmented_dir = Path(args.segmented_dir)
    corrected_dir = Path(args.corrected_dir)
    output_dir = Path(args.output_dir)
    debug = args.debug

    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_files = list(corrected_dir.glob("*.srt"))

    for corrected_path in corrected_files:
        seg_path = segmented_dir / corrected_path.name
        if not seg_path.exists():
            print(f"WARNING: No matching file in segmented for '{corrected_path.name}'. Skipping.")
            continue

        corrected_units = parse_srt(corrected_path)
        seg_units = parse_srt(seg_path)

        # Build a set of normalized texts from corrected
        used_corrected = set()
        units_out = []
        for s_unit in seg_units:
            s_text_norm = normalize_text(s_unit[3])
            match = None
            for idx, c_unit in enumerate(corrected_units):
                c_text_norm = normalize_text(c_unit[3])
                if c_text_norm == s_text_norm and idx not in used_corrected:
                    match = c_unit
                    used_corrected.add(idx)
                    break
            if match:
                units_out.append(match)
            else:
                num, start, end, text = s_unit
                units_out.append((num, start, end, text + " [NOT-SIGNED]"))

        # SANITY CHECK
        if len(units_out) != len(seg_units):
            print(f"ERROR: Output subtitle count {len(units_out)} does not match segmented count {len(seg_units)} for '{corrected_path.name}'. Skipping writing this file.")
            continue

        not_signed_count = sum(1 for unit in units_out if "[NOT-SIGNED]" in unit[3])

        out_path = output_dir / corrected_path.name
        write_srt(units_out, out_path, debug=debug)
        print(f"Processed {corrected_path.name}: {len(units_out)} units, {not_signed_count} were [NOT-SIGNED].")

if __name__ == "__main__":
    main()
