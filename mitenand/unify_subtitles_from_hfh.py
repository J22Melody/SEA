#!/usr/bin/env python3

import argparse
import shutil
import re
from pathlib import Path
import pympi

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ELAN EAF subtitles to SRT using matching rules."
    )
    parser.add_argument(
        "--corrected_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/elan_corrected_hfh_students",
        help="Path to corrected ELAN files (source)."
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/elan_segmented",
        help="Path to original ELAN files (reference for matching)."
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_corrected_final",
        help="Destination for generated SRT files."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: Only print actions, do not write files."
    )
    return parser.parse_args()

def extract_title(filename):
    """Removes leading date and returns the title part, lowercased."""
    return re.sub(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_", "", filename.lower())

def find_matching_originals(corrected_path, original_files):
    """Return a list of original files whose full (date+title) name is a subset of corrected's name."""
    corrected_name = corrected_path.stem.lower()
    matches = []
    for orig in original_files:
        orig_name = orig.stem.lower()
        if orig_name in corrected_name:
            matches.append(orig)
    return matches

def get_subtitle_tier(eaf_obj):
    """Return tier name for various spellings of subtitle corrected."""
    # List all possible correct and typo forms (all lower)
    possible_tiers = [
        "subtitle corrected",
        "subtitle_corrected",
        "subtitle correct",
        "subtitel_corrected",
        "subtitel correct",
        "subtitle_correc",
        "subtitel corrected",
        "subtitled corrected",
        "subtitle_corr",
    ]
    # Build a dict: lowercase -> real tier name in file
    tier_map = {tier.lower(): tier for tier in eaf_obj.tiers}
    for wanted in possible_tiers:
        if wanted in tier_map:
            return tier_map[wanted]
    return None

def ms_to_srt_time(ms):
    """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def elan_to_srt(eaf_path, srt_path, debug=False):
    """Convert ELAN EAF file to SRT, extracting subtitle-corrected tier with backup logic."""
    eaf_obj = pympi.Elan.Eaf(str(eaf_path))
    tier_name = get_subtitle_tier(eaf_obj)
    if not tier_name:
        print(f"  ERROR: No subtitle-corrected tier in '{eaf_path.name}', skipping.")
        return False

    annotations = eaf_obj.get_annotation_data_for_tier(tier_name)
    if not annotations:
        print(f"  WARNING: No annotations in {tier_name} tier of '{eaf_path.name}', skipping.")
        return False

    # Check if all values are empty
    if all((value is None or not value.strip()) for (start, end, value) in annotations):
        # Try to use the SUBTITLE tier if possible
        backup_tier = None
        for t in eaf_obj.tiers:
            if t.strip().lower() == "subtitle":
                backup_tier = t
                break
        if backup_tier is not None:
            backup_annotations = eaf_obj.get_annotation_data_for_tier(backup_tier)
            if len(backup_annotations) == len(annotations):
                print(f"  INFO: Using SUBTITLE tier as backup for '{eaf_path.name}'.")
                # Replace only the text (timings stay from the original tier, but usually they are the same)
                annotations = [
                    (orig[0], orig[1], backup[2]) for orig, backup in zip(annotations, backup_annotations)
                ]
            else:
                print(f"  ERROR: Backup tier 'SUBTITLE' found but annotation count does not match ({len(backup_annotations)} vs {len(annotations)}), skipping '{eaf_path.name}'.")
                return False
        else:
            print(f"  ERROR: All annotations empty and no 'SUBTITLE' tier found in '{eaf_path.name}', skipping.")
            return False

    lines = []
    for idx, (start, end, value) in enumerate(annotations, start=1):
        start_srt = ms_to_srt_time(start)
        end_srt = ms_to_srt_time(end)
        text = value.strip() if value else ""
        lines.append(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n")  # this \n is the empty line after the subtitle

    # Join all lines with a real empty line between subtitles
    srt_content = "\n".join(lines)

    if debug:
        print(f"[DEBUG] Would write SRT with {len(lines)} subtitles to '{srt_path.name}'")
    else:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f"Wrote SRT '{srt_path.name}' with {len(lines)} subtitles.")
    return True

def main():
    args = parse_args()

    corrected_dir = Path(args.corrected_dir)
    original_dir = Path(args.original_dir)
    final_dir = Path(args.final_dir)
    debug = args.debug

    if not final_dir.exists():
        final_dir.mkdir(parents=True)

    original_files = list(original_dir.glob("*.eaf"))
    matched_count = 0
    unmatched_count = 0

    for corrected_path in corrected_dir.glob("*.eaf"):
        matches = find_matching_originals(corrected_path, original_files)
        if len(matches) == 1:
            dest_name = matches[0].stem + ".srt"  # Use original name, .srt
            dest_path = final_dir / dest_name
            print(f"Converting '{corrected_path.name}' to '{dest_path.name}' (using original name)")
            ok = elan_to_srt(corrected_path, dest_path, debug=debug)
            if ok:
                matched_count += 1
            else:
                unmatched_count += 1
        else:
            if len(matches) > 1:
                print(f"WARNING: Multiple matches for '{corrected_path.name}': {[m.name for m in matches]}. Skipping.")
            else:
                print(f"No match for '{corrected_path.name}' in original_dir. Skipping.")
            unmatched_count += 1

    print(f"\nSummary:")
    print(f"Matched (converted): {matched_count}")
    print(f"Unmatched (skipped): {unmatched_count}")

if __name__ == "__main__":
    main()
