import re
import os
import sys
import bisect
import glob
from io import StringIO
import pysrt
import webvtt
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def timestamp_to_seconds(time_str: str) -> float:
    """
    Convert a timestamp in the format "HH:MM:SS.mmm" to total seconds (float).
    """
    parts = time_str.strip().split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split('.')
    seconds = int(sec_parts[0])
    millis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

def seconds_to_timestamp(total_seconds: float) -> str:
    """
    Convert total seconds (float) to a timestamp string "HH:MM:SS.mmm".
    """
    if total_seconds < 0:
        total_seconds = 0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def fix_vtt_format(vtt_content: str) -> str:
    """
    Fix cues that do not follow the standard two-line format.
    If a line has two timestamps without the "-->" delimiter and text on the same line,
    reformat it into a proper cue.
    """
    lines = vtt_content.splitlines()
    if not lines:
        return vtt_content
    # Ensure header exists
    if not lines[0].strip().startswith("WEBVTT"):
        lines.insert(0, "WEBVTT")
    new_lines = [lines[0]]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            new_lines.append("")
            continue
        if "-->" in line:
            new_lines.append(line)
            continue
        pattern = r"^(\d{2}:\d{2}:\d{2}\.\d{3})\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+(.*)$"
        match = re.match(pattern, line)
        if match:
            t1, t2, text = match.groups()
            new_lines.append(f"{t1} --> {t2}")
            new_lines.append(text)
            new_lines.append("")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def ensure_valid_vtt_format(vtt_content: str) -> str:
    """
    Ensure that the VTT content has a valid header and fixed cue format.
    """
    vtt_content = vtt_content.strip()
    if not vtt_content.startswith("WEBVTT"):
        vtt_content = "WEBVTT\n\n" + vtt_content
    return fix_vtt_format(vtt_content)

def shift_cues(cues, delta_start: float, delta_end: float):
    """
    Shift the start and end times of each cue (a dict with keys "start", "end", and "mid")
    by delta_start and delta_end seconds, respectively.
    """
    for cue in cues:
        cue["start"] += delta_start
        cue["end"] += delta_end
        cue["mid"] = (cue["start"] + cue["end"]) / 2
    return cues

def get_subtitle_cues(path: str):
    """
    Parse .vtt or .srt subtitle file and return (header_lines, cues), where:
      - header_lines: ["WEBVTT"] or ["SRT"]
      - cues: list of dicts with 'start', 'end', 'mid', 'text'
    """
    def timestamp_to_seconds(ts: str) -> float:
        ts = ts.replace(',', '.')
        h, m, s = ts.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    ext = Path(path).suffix.lower()
    header_lines = ["SRT"] if ext == ".srt" else ["WEBVTT"]
    cues = []

    try:
        if ext == ".srt":
            subs = pysrt.open(path)
            for sub in subs:
                start = sub.start.ordinal / 1000.0
                end = sub.end.ordinal / 1000.0
                text = " ".join(sub.text.splitlines()).strip()
                cues.append({
                    'start': start,
                    'end': end,
                    'mid': (start + end) / 2,
                    'text': text
                })
        else:
            # FIXME: suppress the prints
            vtt_obj = webvtt.read(path)
            for cue in vtt_obj:
                text = cue.text.strip()
                start_sec = timestamp_to_seconds(cue.start)
                end_sec = timestamp_to_seconds(cue.end)
                cues.append({
                    'start': start_sec,
                    'end': end_sec,
                    'mid': (start_sec + end_sec) / 2,
                    'text': text
                })
    except Exception as e:
        print(f"Error reading subtitle file {path}: {e}")
    
    return header_lines, cues

def reconstruct_vtt(header_lines, cues) -> str:
    """
    Reconstruct a VTT file using the standard two-line format:
      1. A header ("WEBVTT")
      2. A blank line
      3. For each cue:
         - A timing line: "HH:MM:SS.mmm --> HH:MM:SS.mmm"
         - A text line
         - A blank line
    """
    output_lines = []
    if not header_lines or header_lines[0].strip() != "WEBVTT":
        output_lines.append("WEBVTT")
    else:
        output_lines.append(header_lines[0])
    output_lines.append("")
    for cue in cues:
        start_timestamp = seconds_to_timestamp(cue['start'])
        end_timestamp = seconds_to_timestamp(cue['end'])
        output_lines.append(f"{start_timestamp} --> {end_timestamp}")
        output_lines.append(cue['text'])
        output_lines.append("")
    return "\n".join(output_lines)

def get_sign_segments_from_eaf(segmentation_file):
    """Parse an ELAN (.eaf) file and return all segments from the SIGN tier."""
    segments = []
    try:
        tree = ET.parse(segmentation_file)
        root = tree.getroot()
    except Exception:
        return segments
    time_order = root.find("TIME_ORDER")
    time_slots = {}
    if time_order is not None:
        for ts in time_order.findall("TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_value = ts.get("TIME_VALUE")
            if ts_value is not None:
                try:
                    time_slots[ts_id] = float(ts_value) / 1000.0
                except ValueError:
                    time_slots[ts_id] = None
    else:
        return segments
    sign_tier = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == "SIGN":
            sign_tier = tier
            break
    if sign_tier is None:
        return segments
    for annotation in sign_tier.findall("ANNOTATION"):
        annotation_elem = next(iter(annotation), None)
        if annotation_elem is None:
            continue
        text_elem = annotation_elem.find("ANNOTATION_VALUE")
        text = text_elem.text if text_elem is not None else ""
        start_time = None
        end_time = None
        if "TIME_SLOT_REF1" in annotation_elem.attrib and "TIME_SLOT_REF2" in annotation_elem.attrib:
            ts1 = annotation_elem.attrib["TIME_SLOT_REF1"]
            ts2 = annotation_elem.attrib["TIME_SLOT_REF2"]
            start_time = time_slots.get(ts1, None)
            end_time = time_slots.get(ts2, None)
        mid = (start_time + end_time) / 2 if start_time is not None and end_time is not None else None
        segments.append({'start': start_time, 'end': end_time, 'mid': mid, 'text': text})
    return segments

def write_updated_eaf(eaf_file, cues, video_id, signs=None, additional_signs={}):
    """
    Write an updated ELAN file with new tiers:
      - SIGN_MERGED: merged sign annotations (from `signs`)
      - Additional tiers: for each key in `additional_signs` (if its value is a list), write the sign annotations.
      - SUBTITLE_SHIFTED: DP-aligned subtitle cues.
    The updated file is saved with the suffix "_updated.eaf".
    """
    try:
        tree = ET.parse(eaf_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing ELAN file {eaf_file}: {e}")
        return
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        print(f"No TIME_ORDER element found in {eaf_file}")
        return

    # Process the standard signs tier, if provided.
    if signs:
        sign_tier = ET.Element("TIER", {"TIER_ID": "SIGN_MERGED", "LINGUISTIC_TYPE_REF": "default-lt"})
        for i, sign in enumerate(signs):
            ts1 = f"SIGN_MERGED_TS_{video_id}_{i}_1"
            ts2 = f"SIGN_MERGED_TS_{video_id}_{i}_2"
            new_ts1 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts1, "TIME_VALUE": str(int(sign['start'] * 1000))})
            new_ts2 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts2, "TIME_VALUE": str(int(sign['end'] * 1000))})
            time_order.append(new_ts1)
            time_order.append(new_ts2)
            annotation = ET.Element("ANNOTATION")
            alignable = ET.Element("ALIGNABLE_ANNOTATION", {
                "ANNOTATION_ID": f"a_sign_merged_{video_id}_{i}",
                "TIME_SLOT_REF1": ts1,
                "TIME_SLOT_REF2": ts2
            })
            annotation_value = ET.Element("ANNOTATION_VALUE")
            annotation_value.text = sign['text']
            alignable.append(annotation_value)
            annotation.append(alignable)
            sign_tier.append(annotation)
        root.append(sign_tier)

    # Process additional_signs dictionary for extra tiers.
    if additional_signs:
        for tier_key, tier_signs in additional_signs.items():
            if isinstance(tier_signs, list):
                tier_element = ET.Element("TIER", {"TIER_ID": tier_key, "LINGUISTIC_TYPE_REF": "default-lt"})
                for i, sign in enumerate(tier_signs):
                    ts1 = f"{tier_key}_TS_{video_id}_{i}_1"
                    ts2 = f"{tier_key}_TS_{video_id}_{i}_2"
                    new_ts1 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts1, "TIME_VALUE": str(int(sign['start'] * 1000))})
                    new_ts2 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts2, "TIME_VALUE": str(int(sign['end'] * 1000))})
                    time_order.append(new_ts1)
                    time_order.append(new_ts2)
                    annotation = ET.Element("ANNOTATION")
                    alignable = ET.Element("ALIGNABLE_ANNOTATION", {
                        "ANNOTATION_ID": f"a_{tier_key}_{video_id}_{i}",
                        "TIME_SLOT_REF1": ts1,
                        "TIME_SLOT_REF2": ts2
                    })
                    annotation_value = ET.Element("ANNOTATION_VALUE")
                    annotation_value.text = sign['text']
                    alignable.append(annotation_value)
                    annotation.append(alignable)
                    tier_element.append(annotation)
                root.append(tier_element)
            # else:
            #     print(f"Value for tier '{tier_key}' is not a list, skipping.")

    # Process the subtitle cues.
    subtitle_tier = ET.Element("TIER", {"TIER_ID": "SUBTITLE_SHIFTED", "LINGUISTIC_TYPE_REF": "default-lt"})
    for i, cue in enumerate(cues):
        ts1 = f"SUBTITLE_TS_{video_id}_{i}_1"
        ts2 = f"SUBTITLE_TS_{video_id}_{i}_2"
        new_ts1 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts1, "TIME_VALUE": str(int(cue['start'] * 1000))})
        new_ts2 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts2, "TIME_VALUE": str(int(cue['end'] * 1000))})
        time_order.append(new_ts1)
        time_order.append(new_ts2)
        annotation = ET.Element("ANNOTATION")
        alignable = ET.Element("ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a_subtitle_{video_id}_{i}",
            "TIME_SLOT_REF1": ts1,
            "TIME_SLOT_REF2": ts2
        })
        annotation_value = ET.Element("ANNOTATION_VALUE")
        annotation_value.text = cue['text']
        alignable.append(annotation_value)
        annotation.append(alignable)
        subtitle_tier.append(annotation)
    root.append(subtitle_tier)

    output_eaf = os.path.splitext(eaf_file)[0] + "_updated.eaf"
    tree.write(output_eaf, encoding="utf-8", xml_declaration=True)
    print(f"Written updated ELAN file to {output_eaf}")

def print_results(eval_output, column_names=None):
    """
    Parses evaluation output(s) and prints a formatted table of results.
    
    If eval_output is a string, a single-column table is printed.
    If eval_output is a list, each element is treated as a separate column.
    
    Each evaluation output is expected to have lines like:
    
    total  1172902 subs 9168
    Mean and median start offset: 0.37 / 0.03
    Mean and median end offset: -0.37 / -0.45
    Mean and median start offset (abs): 1.30 / 0.61
    Mean and median end offset (abs): 1.37 / 0.85
    Computed over 1172902 frames, 9168 sentences - Frame-level accuracy: 75.43 F1@0.10: 82.36 F1@0.25: 77.17 F1@0.50: 61.43
    """
    # If eval_output is a string, convert to a single-element list.
    if isinstance(eval_output, str):
        eval_outputs = [eval_output]
        if column_names is None:
            column_names = ["Result"]
    else:
        eval_outputs = eval_output
        if column_names is None:
            column_names = [f"Result {i+1}" for i in range(len(eval_outputs))]
    
    # Parse each evaluation output into a dictionary of metric->value.
    def parse_eval(e):
        d = {}
        for line in e.splitlines():
            line = line.strip()
            if not line:
                continue
            if " - " in line:
                prefix, suffix = line.split(" - ", 1)
                m = re.search(r'over\s+(\d+)\s+frames,\s+(\d+)\s+sentences', prefix)
                if m:
                    d["Total frames"] = m.group(1)
                    d["Total sentences"] = m.group(2)
                pairs = re.findall(r'([A-Za-z0-9@\(\)\-\s]+):\s*([-\d\.\/]+)', suffix)
                for key, value in pairs:
                    key = key.strip()
                    if key in {"F1@10", "10"}:
                        key = "F1@0.10"
                    elif key in {"F1@25", "25"}:
                        key = "F1@0.25"
                    elif key in {"F1@50", "50"}:
                        key = "F1@0.50"
                    d[key] = value.strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                d[key.strip()] = value.strip()
            else:
                tokens = line.split()
                if len(tokens) >= 4 and tokens[0].lower() == "total":
                    d["Total frames"] = tokens[1]
                    d["Total subtitles"] = tokens[3]
                else:
                    d[line.strip()] = ""
        return d

    dicts = [parse_eval(e) for e in eval_outputs]
    
    # Get union of all keys.
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    # Sort keys by: Total frames, Total sentences/subtitles, non-abs offset values,
    # then absolute offsets, frame-level accuracy, then F1 metrics.
    def sort_key(metric):
        m = metric.lower()
        if m.startswith("total frames"):
            return (0, m)
        elif "total sentences" in m or "total subtitles" in m:
            return (1, m)
        elif "start offset" in m and "(abs)" not in m:
            return (2, m)
        elif "end offset" in m and "(abs)" not in m:
            return (3, m)
        elif "start offset" in m and "(abs)" in m:
            return (4, m)
        elif "end offset" in m and "(abs)" in m:
            return (5, m)
        elif m.startswith("frame-level accuracy"):
            return (6, m)
        elif m.startswith("f1@0.10"):
            return (7, m)
        elif m.startswith("f1@0.25"):
            return (8, m)
        elif m.startswith("f1@0.50"):
            return (9, m)
        else:
            return (10, m)
    sorted_keys = sorted(all_keys, key=sort_key)
    
    # Prepare table columns: first column is "Metric", then one column per evaluation output.
    num_columns = 1 + len(dicts)
    headers = ["Metric"] + column_names
    
    # Determine column widths.
    col_widths = []
    first_col_width = max(len("Metric"), max((len(k) for k in sorted_keys), default=0))
    col_widths.append(first_col_width)
    for i in range(len(dicts)):
        header_len = len(column_names[i])
        max_val_len = header_len
        for key in sorted_keys:
            val = dicts[i].get(key, "")
            if len(val) > max_val_len:
                max_val_len = len(val)
        col_widths.append(max_val_len)
    
    # Print header row.
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    separator_row = "-+-".join("-" * col_widths[i] for i in range(num_columns))
    print(separator_row)
    
    # Print each metric row.
    for key in sorted_keys:
        row = [key.ljust(col_widths[0])]
        for i in range(len(dicts)):
            val = dicts[i].get(key, "")
            row.append(val.ljust(col_widths[i+1]))
        print(" | ".join(row))

def extract_f1_score(eval_output):
    """Extract F1@0.50 score from evaluation output."""
    m = re.search(r"F1@0\.50:\s*([\d.]+)", eval_output)
    return float(m.group(1)) if m else 0.0
    
def get_cslr_signs(video_id, cslr_dir):
    """
    Recursively search for a CSV file named <video_id>.csv under cslr_dir
    and return sign annotations.
    """
    pattern = os.path.join(cslr_dir, '**', f"{video_id}.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return []
    csv_file = files[0]
    signs = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            approx_gloss = row.get("approx gloss sequence", "")
            english_sentence = row.get("english sentence", "").strip()
            matches = re.findall(r'(\S+)\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]', approx_gloss)
            for word, start_str, end_str in matches:
                start = float(start_str)
                end = float(end_str)
                mid = (start + end) / 2
                signs.append({'start': start, 'end': end, 'mid': mid, 'text': word, 'subtitle': english_sentence})
    return sorted(signs, key=lambda s: s['start'])

def get_cmpl_signs(video_id, segmentation_dir_cmpl):
    """
    Read sign segments from a directory in CMPL format.
    In this format, segmentation_dir_cmpl contains subdirectories for each video id,
    and inside each subdirectory there is a demo.vtt file.
    Each subtitle cue in the vtt file represents a sign segment (with an empty text).
    """
    vtt_path = os.path.join(segmentation_dir_cmpl, video_id, "demo.vtt")
    if not os.path.exists(vtt_path):
        return []
    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            vtt_content = f.read()
    except Exception:
        return []
    header_lines, cues = get_subtitle_cues(vtt_content)
    segments = []
    for cue in cues:
        segments.append({'start': cue['start'], 'end': cue['end'], 'mid': (cue['start']+cue['end'])/2, 'text': ""})
    return segments

def get_pseudo_signs(video_id, pseudo_glosses_dir):
    """
    Read pseudo gloss annotations for a given video_id from a CSV file located at:
        <pseudo_glosses_dir>/<video_id>.csv
    Each row in the CSV is expected to have columns: start, end, text, probs.
    
    Returns a list of dictionaries, where each dictionary contains:
        'start': float, 
        'end': float, 
        'mid': float (average of start and end),
        'text': string, 
        'probs': float
    """
    csv_path = os.path.join(pseudo_glosses_dir, f"{video_id}.csv")
    signs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row.get("start", 0))
                end = float(row.get("end", 0))
                text = row.get("text", "")
                probs = float(row.get("probs", 0))
                mid = (start + end) / 2.0
                signs.append({
                    "start": start,
                    "end": end,
                    "mid": mid,
                    "text": text,
                    "probs": probs
                })
            except Exception as e:
                # Optionally log the error.
                continue
    return signs

def merge_signs(elan_signs, new_signs, conservative=True, overlapIoU=0.2):
    """
    Merge additional signs into the base ELAN sign list.
    For each new sign, remove overlapping ELAN segments and insert the new sign.
    If conservative is True, the new sign is only appended if it overlaps with at least one segment in elan_signs
    with an Intersection over Union (IoU) greater than overlapIoU.
    If conservative is False, the new sign is always appended.
    """
    def iou(interval1, interval2):
        # Calculate Intersection over Union (IoU) for two intervals.
        inter = max(0, min(interval1['end'], interval2['end']) - max(interval1['start'], interval2['start']))
        union = max(interval1['end'], interval2['end']) - min(interval1['start'], interval2['start'])
        return inter / union if union > 0 else 0

    # Sort the input lists.
    elan_signs = sorted(elan_signs, key=lambda s: s['start'])
    new_signs = sorted(new_signs, key=lambda s: s['start'])
    
    # Cache the IDs of the original elan_signs for fast membership checking.
    original_ids = {id(seg) for seg in elan_signs}
    
    # Start with the original ELAN signs.
    merged = elan_signs[:]
    
    for ns in new_signs:
        if conservative:
            # Only merge ns if it overlaps with at least one original segment with IoU > overlapIoU.
            if not any(
                iou(seg, ns) > overlapIoU
                for seg in merged
                if seg['end'] > ns['start'] and seg['start'] < ns['end'] and id(seg) in original_ids
            ):
                continue

        # Remove overlapping segments from the original elan_signs.
        new_merged = [seg for seg in merged
                      if not (id(seg) in original_ids and seg['end'] > ns['start'] and seg['start'] < ns['end'])]
        # Insert the new sign into the sorted list using bisect.
        insert_points = [s['start'] for s in new_merged]
        idx = bisect.bisect_left(insert_points, ns['start'])
        new_merged.insert(idx, ns)
        merged = new_merged

    return merged

def filter_cues_by_cslr(cues, cslr_signs):
    """
    Return a filtered list of cues that have temporal overlap with any of the CSLR signs.
    A cue overlaps if its end time is greater than a sign's start time and its start time is less than the sign's end time.
    """
    filtered_cues = []
    for cue in cues:
        overlap = False
        for sign in cslr_signs:
            if cue['end'] > sign['start'] and cue['start'] < sign['end']:
                filtered_cues.append(cue)
                overlap = True
                break
        if not overlap:
            cue['text'] = cue['text'] + ' {CSLR_EXCLUDED}'
            filtered_cues.append(cue)
    return filtered_cues

def softmax_normalize(matrix, axis=None, tau=10):
    """
    Apply softmax normalization with temperature tau along the specified axis.
    Always returns a proper softmax (sums to 1).
    """
    exp_vals = np.exp(matrix / tau)
    sum_vals = np.sum(exp_vals, axis=axis, keepdims=True)
    softmax = exp_vals / sum_vals
    return softmax

def zscore_sigmoid_normalize(vector, tau=10):
    """
    Normalize a 1D numpy array using z-score followed by sigmoid.
    Values are mapped to (0, 1).
    """
    vector = np.asarray(vector)
    mean = np.mean(vector)
    std = np.std(vector) + 1e-6  # avoid division by zero
    z = (vector - mean) / std
    return 1 / (1 + np.exp(-z / tau))  # sigmoid with tau controlling sharpness

def sinkhorn_normalize(matrix, num_iters=10, eps=1e-8):
    """
    Apply Sinkhorn normalization to a 2D matrix (or 1 row at a time).
    Returns a doubly stochastic matrix.
    """
    matrix = np.maximum(matrix, eps)  # avoid division by zero
    for _ in range(num_iters):
        matrix /= matrix.sum(axis=1, keepdims=True)
        matrix /= matrix.sum(axis=0, keepdims=True)
    return matrix
