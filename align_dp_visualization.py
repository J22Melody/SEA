import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import softmax_normalize

def visualize_similarity_heatmap(sim_matrix, cues, sign_segments, gt_cues=None, new_cues=None, fps=25, start_time_window=60, end_time_window=100):
    """Visualize similarity matrix with aligned and ground truth cues at frame-level resolution.
    
    [Documentation omitted for brevity]
    """

    def format_time_full(total_seconds):
        if total_seconds < 0:
            total_seconds = 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    time_axis = np.arange(start_time_window, end_time_window + 1/fps, 1/fps)
    seg_indices_in_window = []
    for idx, seg in enumerate(sign_segments):
        if seg['start'] is not None and seg['end'] is not None:
            if seg['end'] >= start_time_window and seg['start'] <= end_time_window:
                seg_indices_in_window.append(idx)

    cue_indices_in_window = []
    for idx, cue in enumerate(cues):
        if cue['start'] is not None and cue['end'] is not None:
            if cue['end'] >= start_time_window and cue['start'] <= end_time_window:
                cue_indices_in_window.append(idx)
    M_filtered = len(cue_indices_in_window)

    heatmap_data = np.full((M_filtered, len(time_axis)), np.nan)
    for j in seg_indices_in_window:
        seg = sign_segments[j]
        if seg['start'] is None or seg['end'] is None:
            continue
        indices = np.where((time_axis >= seg['start']) & (time_axis <= seg['end']))[0]
        if len(indices) > 0:
            for r, cue_idx in enumerate(cue_indices_in_window):
                heatmap_data[r, indices] = sim_matrix[cue_idx, j]

    masks_list = []
    normalized_heatmap = heatmap_data.copy()
    window_size = 40
    filtered_seg_mid_times = []
    for j in seg_indices_in_window:
        seg = sign_segments[j]
        mid_t = (seg['start'] + seg['end']) / 2
        filtered_seg_mid_times.append(mid_t)
    filtered_seg_mid_times = np.array(filtered_seg_mid_times)

    for row_idx, cue_idx in enumerate(cue_indices_in_window):
        cue = cues[cue_idx]
        cue_mid = (cue['start'] + cue['end']) / 2
        diffs = np.abs(filtered_seg_mid_times - cue_mid)
        candidate_order = np.argsort(diffs)[:window_size]
        
        valid_candidates = [c for c in candidate_order if c < len(seg_indices_in_window)]
        mask = np.zeros(len(time_axis), dtype=bool)
        for candidate in valid_candidates:
            seg_global_idx = seg_indices_in_window[candidate]
            seg = sign_segments[seg_global_idx]
            if seg['start'] is None or seg['end'] is None:
                continue
            candidate_mask = (time_axis >= seg['start']) & (time_axis <= seg['end'])
            mask = mask | candidate_mask
        masks_list.append(mask)
        if np.any(mask):
            local_vals = heatmap_data[row_idx, mask]
            # local_vals = softmax_normalize(local_vals, axis=0, tau=10)
            normalized_heatmap[row_idx, mask] = local_vals

    global_min = np.nanmin(normalized_heatmap)
    for row_idx in range(M_filtered):
        mask = masks_list[row_idx]
        normalized_heatmap[row_idx, ~mask] = global_min

    heatmap_data = normalized_heatmap

    duration = end_time_window - start_time_window
    fig_width = duration * 1.5
    fig_width *= 0.3  # cut width by half

    plt.figure(figsize=(fig_width, 10))
    left_frac = min(10 / fig_width, 0.9)  # clamp to keep it valid
    plt.subplots_adjust(left=left_frac)
    im = plt.imshow(heatmap_data, aspect='auto', origin='upper',
                interpolation='nearest', cmap='YlGnBu',
                extent=(start_time_window, end_time_window, M_filtered, 0))

    # No colorbar

    first_tick = int(np.ceil(start_time_window / 5.0) * 5)
    last_tick = int(np.floor(end_time_window / 5.0) * 5)
    tick_positions = np.arange(first_tick, last_tick + 1, 5)
    plt.xticks(tick_positions, [format_time_full(t) for t in tick_positions], rotation=45)

    # Build compact y-axis labels: full words, capped at 12 chars, add "..." if truncated
    def _truncate_full_words(text, limit=12):
        text = " ".join((text or "").split())  # collapse whitespace
        if not text:
            return "..."
        words = text.split(" ")
        label = ""
        for w in words:
            candidate = w if not label else f"{label} {w}"
            if len(candidate) <= limit:
                label = candidate
            else:
                break
        if not label:  # first word longer than limit
            label = words[0][:limit]
        if len(text) > len(label):
            label += "..."
        return label

    y_labels = []
    for idx in cue_indices_in_window:
        cue = cues[idx]
        text = (cue.get('text') or "")
        y_labels.append(_truncate_full_words(text, limit=12))

    plt.yticks(np.arange(0.5, M_filtered + 0.5), y_labels)
    # Increase font sizes for better visibility
    plt.tick_params(axis='x', labelsize=16, pad=4)
    plt.tick_params(axis='y', labelsize=16)

    ax = plt.gca()
    # Ensure axis labels and title are removed (explicit empty strings)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    plt.suptitle("")

    if gt_cues:
        gt_text_map = {}
        for gt_cue in gt_cues:
            if gt_cue['text']:
                clean_text = gt_cue['text'].strip().replace("\n", " ")[:50]
                gt_text_map[clean_text] = gt_cue
        
        for i, cue_idx in enumerate(cue_indices_in_window[::-1]):
            cue = cues[cue_idx]
            clean_text = cue['text'].strip().replace("\n", " ")[:50]
            gt_cue = gt_text_map.get(clean_text)
            if gt_cue and gt_cue['start'] and gt_cue['end']:
                box_start = max(gt_cue['start'], start_time_window)
                box_end = min(gt_cue['end'], end_time_window)
                if box_end > box_start:
                    y_box = M_filtered - i - 1
                    rect = plt.Rectangle((box_start, y_box), box_end-box_start, 1,
                                         edgecolor='#ff00ff', facecolor='none',
                                         linewidth=3)
                    ax.add_patch(rect)

    # New block for new_cues (dashed, different high-contrast color)
    if new_cues:
        new_text_map = {}
        for new_cue in new_cues:
            if new_cue['text']:
                clean_text = new_cue['text'].strip().replace("\n", " ")[:50]
                new_text_map[clean_text] = new_cue
        
        for i, cue_idx in enumerate(cue_indices_in_window[::-1]):
            cue = cues[cue_idx]
            clean_text = (cue['text'] or "").strip().replace("\n", " ")[:50]
            new_cue = new_text_map.get(clean_text)
            if new_cue and new_cue['start'] and new_cue['end']:
                box_start = max(new_cue['start'], start_time_window)
                box_end = min(new_cue['end'], end_time_window)
                if box_end > box_start:
                    y_box = M_filtered - i - 1
                    rect = plt.Rectangle((box_start, y_box), box_end-box_start, 1,
                                         edgecolor='#ffa500', facecolor='none',
                                         linewidth=3, linestyle='--')
                    ax.add_patch(rect)

    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "heatmap.png"))
    plt.close()
    print("Saved heatmap to heatmap.png")
