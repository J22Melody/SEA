#!/usr/bin/env python3

import argparse
import shutil
import re
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unify and copy corrected subtitles if matching originals exist (ignoring leading dates)."
    )
    parser.add_argument(
        "--corrected_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_corrected_swisstxt_segmented",
        help="Path to corrected segmented subtitles (source)."
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_segmented",
        help="Path to original segmented subtitles (reference for matching)."
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default="/shares/iict-sp2.ebling.cl.uzh/Deliverable/DSGS/mitenand/subtitles_corrected_final",
        help="Destination for unified/corrected subtitles."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: Only print actions, do not copy files."
    )
    return parser.parse_args()

def extract_title(filename):
    """Removes leading date and returns the title part, lowercased."""
    # Removes everything up to and including the first underscore after a date
    return re.sub(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_", "", filename.lower())

def find_matching_originals(corrected_path, original_files):
    """Return a list of original files whose title is a subset of corrected's title (ignoring date)."""
    corrected_title = extract_title(corrected_path.stem)
    matches = []
    for orig in original_files:
        orig_title = extract_title(orig.stem)
        if orig_title in corrected_title:
            matches.append(orig)
    return matches

def main():
    args = parse_args()
    
    corrected_dir = Path(args.corrected_dir)
    original_dir = Path(args.original_dir)
    final_dir = Path(args.final_dir)
    debug = args.debug

    if not final_dir.exists():
        final_dir.mkdir(parents=True)

    original_files = list(original_dir.glob("*.srt"))
    matched_count = 0
    unmatched_count = 0

    for corrected_path in corrected_dir.glob("*.srt"):
        matches = find_matching_originals(corrected_path, original_files)
        if len(matches) == 1:
            dest_path = final_dir / matches[0].name  # Use original name
            if debug:
                print(f"[DEBUG] Would copy '{corrected_path.name}' to '{dest_path.name}' (using original name)")
            else:
                print(f"Copying '{corrected_path.name}' to '{dest_path.name}' (using original name)")
                shutil.copy2(str(corrected_path), str(dest_path))
            matched_count += 1
        else:
            if len(matches) > 1:
                print(f"WARNING: Multiple matches for '{corrected_path.name}': {[m.name for m in matches]}. Skipping.")
            else:
                print(f"No match for '{corrected_path.name}' in original_dir. Skipping.")
            unmatched_count += 1

    print(f"\nSummary:")
    print(f"Matched (copied): {matched_count}")
    print(f"Unmatched (skipped): {unmatched_count}")

if __name__ == "__main__":
    main()
