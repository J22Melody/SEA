import copy
import numpy as np
from tqdm import tqdm
from numba import njit

# --- DP alignment helper functions ---

def compute_gap_cost(sign_segments):
    """
    Vectorized computation of the gap cost matrix.
    
    Given a list of sign segments (each a dict with 'start' and 'end'),
    compute a (N+1)x(N+1) matrix where for i < j:
        gap_cost[i][j] = sum_{p=i+1}^{j} max(0, sign_segments[p]['start'] - sign_segments[p-1]['end'])
    and gap_cost[i][j] = 0 for j <= i.
    """
    N = len(sign_segments)
    # Build arrays of start and end times.
    starts = np.array([seg['start'] for seg in sign_segments])
    ends = np.array([seg['end'] for seg in sign_segments])
    # Compute the gap between adjacent segments and clip negative values to zero.
    gaps = np.maximum(0, starts[1:] - ends[:-1])
    # Compute cumulative sum of gaps. This yields an array of length N.
    cumsum = np.concatenate(([0], np.cumsum(gaps)))
    # Now, gap_cost for indices 0 <= i,j < N is: cumsum[j] - cumsum[i] (for j>=i)
    gap_cost = cumsum.reshape(1, -1) - cumsum.reshape(-1, 1)
    # For j < i, the difference will be negative; we set these to zero.
    gap_cost[gap_cost < 0] = 0
    # Pad to shape (N+1, N+1) if desired.
    gap_cost_padded = np.zeros((N+1, N+1))
    gap_cost_padded[:N, :N] = gap_cost
    return gap_cost_padded

def compute_alignment_cost(cue_start, cue_end, group_start, group_end, 
                           duration_penalty_weight, gap_penalty_weight, gap,
                           similarity_total, similarity_weight):
    """
    Compute the alignment cost between a cue and a group of sign segments.
    
    The cost comprises differences in start/end times, duration differences,
    a gap penalty, and a similarity penalty (where a higher similarity_total reduces the cost).
    """
    cue_duration = cue_end - cue_start
    group_duration = group_end - group_start
    return (abs(cue_start - group_start) +
            abs(cue_end - group_end) +
            duration_penalty_weight * abs(cue_duration - group_duration) +
            gap_penalty_weight * gap +
            similarity_weight * (- similarity_total))

def cost_for_subgroup(subgroup, original_start, original_end, group_global_start, cue_index,
                      duration_penalty_weight, gap_penalty_weight, similarity_weight,
                      sim_cumsum=None):
    """
    Helper function for post-processing cost computation for a candidate subgroup.
    
    Computes the alignment cost for a subgroup of sign segments.
    """
    if not subgroup:
        return float('inf')
    subgroup_start = subgroup[0]['start']
    subgroup_end = subgroup[-1]['end']
    gap_val = sum(subgroup[i]['start'] - subgroup[i-1]['end'] for i in range(1, len(subgroup)))
    
    similarity_total_candidate = 0
    if sim_cumsum is not None and cue_index > 0:
        L = len(subgroup)
        similarity_total_candidate = sim_cumsum[cue_index-1, group_global_start + L] - sim_cumsum[cue_index-1, group_global_start]
    
    return compute_alignment_cost(
        original_start, original_end,
        subgroup_start, subgroup_end,
        duration_penalty_weight, gap_penalty_weight, gap_val,
        similarity_total_candidate, similarity_weight
    )

@njit
def dp_inner_loop(M, N, dp, prev, cue_starts, cue_ends, sign_starts, sign_ends, gap_cost,
                  candidate_min, candidate_max, sim_matrix, duration_penalty_weight,
                  gap_penalty_weight, similarity_weight, use_similarity):
    """
    JIT-compiled inner loop for dynamic programming.
    
    For each cue i (1-indexed in dp), this function:
      - Determines the candidate sign window from candidate_min to candidate_max.
      - If use_similarity is True, it extracts the corresponding slice from sim_matrix,
        applies softmax normalization via softmax_normalize_jit (with default axis and tau),
        and computes the cumulative sum.
      - Then iterates over j and k to update dp and prev.
    """
    for i in range(1, M+1):
        cand_min = candidate_min[i-1]
        cand_max = candidate_max[i-1]
        if i > cand_min + 1:
            j_lower = i
        else:
            j_lower = cand_min + 1
        if cand_max + 1 < N:
            j_upper = cand_max + 1
        else:
            j_upper = N

        if use_similarity:
            length = cand_max - cand_min + 1
            local_sim = np.empty(length, dtype=dp.dtype)
            for idx in range(length):
                local_sim[idx] = sim_matrix[i-1, cand_min + idx]
            # Use the JIT softmax normalization on the row (local_sim)
            # local_sim = softmax_normalize_jit(local_sim)
            local_sim_cumsum = np.empty(length + 1, dtype=dp.dtype)
            local_sim_cumsum[0] = 0.0
            for idx in range(length):
                local_sim_cumsum[idx+1] = local_sim_cumsum[idx] + local_sim[idx]
        
        for j in range(j_lower, j_upper+1):
            for k in range(i-1, j):
                group_start = sign_starts[k]
                group_end = sign_ends[j-1]
                if k < j-1:
                    total_gap = gap_cost[k, j-1]
                else:
                    total_gap = 0.0
                if use_similarity:
                    local_k = k - cand_min
                    if local_k < 0:
                        local_k = 0
                    local_j = j - cand_min
                    if local_j > length:
                        local_j = length
                    similarity_total = local_sim_cumsum[local_j] - local_sim_cumsum[local_k]
                else:
                    similarity_total = 0.0
                cue_start = cue_starts[i-1]
                cue_end = cue_ends[i-1]
                diff_start = abs(cue_start - group_start)
                diff_end = abs(cue_end - group_end)
                diff_duration = abs((cue_end - cue_start) - (group_end - group_start))
                # FIXME: should use the function compute_alignment_cost
                cost_val = diff_start + diff_end + duration_penalty_weight * diff_duration + gap_penalty_weight * total_gap + similarity_weight * (-similarity_total)
                cur_cost = dp[i-1, k] + cost_val
                if cur_cost < dp[i, j]:
                    dp[i, j] = cur_cost
                    prev[i, j] = k

def dp_align_subtitles_to_signs(cues, sign_segments, gt_cues=None,
                                duration_penalty_weight=0.4, gap_penalty_weight=2.0,
                                window_size=40, max_gap=8.0, similarity_weight=10,
                                sim_matrix=None,
                                visualize_similarity=False):
    """Dynamic programming alignment."""
    M = len(cues)
    N = len(sign_segments)
    if M == 0 or N == 0:
        return

    cues_original = copy.deepcopy(cues)
    original_cue_timings = [(c['start'], c['end']) for c in cues]
    
    # Precompute candidate indices for each cue.
    sign_mids = np.array([(seg['start'] + seg['end'])/2 for seg in sign_segments])
    candidate_min_list = []
    candidate_max_list = []
    for cue in cues:
        cue_mid = (cue['start'] + cue['end']) / 2
        cand = np.argsort(np.abs(sign_mids - cue_mid))[:window_size]
        candidate_min_list.append(int(np.min(cand)))
        candidate_max_list.append(int(np.max(cand)))
    candidate_min_arr = np.array(candidate_min_list)
    candidate_max_arr = np.array(candidate_max_list)
    
    # Build numpy arrays for cue and sign timings.
    cue_starts = np.array([c['start'] for c in cues])
    cue_ends = np.array([c['end'] for c in cues])
    sign_starts = np.array([s['start'] for s in sign_segments])
    sign_ends = np.array([s['end'] for s in sign_segments])
    
    # Initialize DP matrices as numpy arrays.
    dp = np.full((M+1, N+1), np.inf, dtype=np.float64)
    prev = np.full((M+1, N+1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    use_similarity = sim_matrix is not None
    if use_similarity:
        # Compute the cumulative sum along each row.
        sim_cumsum = np.zeros((M, N+1))
        for i in tqdm(range(M), desc="Computing similarity cumulative sum"):
            sim_cumsum[i, 1:] = np.cumsum(sim_matrix[i, :])
    else:
        # HACK: dummy array for the JIT function
        sim_matrix = np.zeros((M, N), dtype=np.float64)

    gap_cost = compute_gap_cost(sign_segments)

    # Call the JIT-compiled DP inner loop.
    dp_inner_loop(M, N, dp, prev, cue_starts, cue_ends, sign_starts, sign_ends, gap_cost,
                  candidate_min_arr, candidate_max_arr, sim_matrix,
                  duration_penalty_weight, gap_penalty_weight, similarity_weight, use_similarity)
    
    # Backtracking to recover boundaries.
    best_j = int(np.argmin(dp[M, :]))
    boundaries = [0] * (M + 1)
    boundaries[M] = best_j
    cur = best_j
    for i in range(M, 0, -1):
        k = int(prev[i, cur])
        boundaries[i - 1] = k
        cur = k

    # Post-processing: refine cue timings by selecting the best candidate subgroup.
    for i in range(M):
        group = sign_segments[boundaries[i]:boundaries[i+1]]
        if not group:
            continue
        original_start, original_end = original_cue_timings[i]
        group_global_start = boundaries[i]
        
        min_cost = np.inf
        best_subgroup = None
        
        current_subgroup = []
        current_subgroup_offset = None
        for j, seg in enumerate(group):
            if not current_subgroup:
                current_subgroup = [seg]
                current_subgroup_offset = j
            else:
                if seg['start'] - group[j-1]['end'] <= max_gap:
                    current_subgroup.append(seg)
                else:
                    candidate_global_start = group_global_start + current_subgroup_offset
                    candidate_cost = cost_for_subgroup(
                        current_subgroup, original_start, original_end,
                        candidate_global_start, i,
                        duration_penalty_weight, gap_penalty_weight, similarity_weight,
                        sim_cumsum if use_similarity else None
                    )
                    if candidate_cost < min_cost:
                        min_cost = candidate_cost
                        best_subgroup = current_subgroup.copy()
                    current_subgroup = [seg]
                    current_subgroup_offset = j
        if current_subgroup:
            candidate_global_start = group_global_start + current_subgroup_offset
            candidate_cost = cost_for_subgroup(
                current_subgroup, original_start, original_end,
                candidate_global_start, i,
                duration_penalty_weight, gap_penalty_weight, similarity_weight,
                sim_cumsum if use_similarity else None
            )
            if candidate_cost < min_cost:
                best_subgroup = current_subgroup
        
        if best_subgroup:
            cues[i]['start'] = best_subgroup[0]['start']
            cues[i]['end'] = best_subgroup[-1]['end']
            cues[i]['mid'] = (cues[i]['start'] + cues[i]['end']) / 2

    if visualize_similarity:
        # Import the visualization function from the separate file.
        from align_dp_visualization import visualize_similarity_heatmap
        visualize_similarity_heatmap(sim_matrix, cues_original, sign_segments, gt_cues, cues)