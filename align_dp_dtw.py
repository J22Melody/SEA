import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw

def dp_align_subtitles_to_signs_dtw(cues, sign_segments):
    """
    A simple DTW-based alignment function.
    
    For each cue and sign segment, we form a tuple (mid, id) where:
      - For a cue, mid = cue['mid'] (or (start+end)/2) and id is a unique numeric ID for cue['text'] (0 if empty).
      - For a sign segment, mid = seg['mid'] (or computed) and id is a unique numeric ID for seg.get('subtitle', '') (0 if empty).
    
    The DTW distance function is defined as follows:
      - If both IDs are nonzero and equal, the distance is 0.
      - If both IDs are nonzero and different, the distance is abs(mid_cue - mid_seg) + 1000.
      - Otherwise, the distance is abs(mid_cue - mid_seg).
      
    After DTW, each cue is updated so that its start is the minimum start and its end is the maximum end among all sign segments assigned to it.
    """
    # Build a set of all non-empty text values from cues and sign segments.
    texts = set()
    for c in cues:
        if c['text']:
            texts.add(c['text'])
    for s in sign_segments:
        text = s.get('subtitle', '')
        if text:
            texts.add(text)
    # Map each non-empty text to a unique ID starting from 1; use 0 for empty.
    text_to_id = {text: idx+1 for idx, text in enumerate(sorted(texts))}
    
    # Build the sequences for DTW.
    cue_seq = [((c['mid'] if isinstance(c['mid'], float) else (c['start'] + c['end'])/2), 
                 text_to_id[c['text']] if c['text'] in text_to_id else 0)
               for c in cues]
    seg_seq = [((s['mid'] if isinstance(s['mid'], float) else (s['start'] + s['end'])/2),
                 text_to_id[s.get('subtitle', '')] if s.get('subtitle', '') in text_to_id else 0)
               for s in sign_segments]
    
    # Define the distance function.
    def dtw_dist(x, y):
        # x = (cue_mid, cue_id), y = (seg_mid, seg_id)
        if x[1] != 0 and y[1] != 0:
            if x[1] == y[1]:
                return 0
            else:
                return abs(x[0] - y[0]) + 1000
        return abs(x[0] - y[0])
    
    # Compute DTW alignment.
    distance, path = fastdtw(cue_seq, seg_seq, dist=dtw_dist)
    
    # Build assignments: cue index -> list of sign segment indices.
    assignments = {i: [] for i in range(len(cues))}
    for i, j in path:
        assignments[i].append(j)
    
    # Update each cue boundaries based on assigned sign segments.
    for i, cue in enumerate(cues):
        if not assignments[i]:
            continue
        assigned_segments = [sign_segments[j] for j in assignments[i]]
        new_start = min(seg['start'] for seg in assigned_segments)
        new_end = max(seg['end'] for seg in assigned_segments)
        cue['start'] = new_start
        cue['end'] = new_end
        cue['mid'] = (new_start + new_end) / 2
