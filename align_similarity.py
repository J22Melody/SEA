import numpy as np

from utils import softmax_normalize, zscore_sigmoid_normalize, sinkhorn_normalize

def compute_similarity_matrix(
    cues, sign_segments, similarity_measure,
    subtitle_embedding=None, subtitle_embedding_tokenized=None,
    segmentation_embedding=None, tokenize_text_embedding=False,
    text_embedding_pooling='max',
    normalize_rowwise=True,
    normalize_columnwise=False,
    normalization_method='softmax',
    filter_threshold: float = 0.0,
):
    """
    Compute a similarity matrix between cues and sign segments along with a cumulative sum.
    
    [Documentation omitted for brevity]
    """
    M = len(cues)
    N = len(sign_segments)
    sim_matrix = None

    if similarity_measure in ["cslr_subtitle", "cslr_text"]:
        sim_matrix = np.zeros((M, N))
        for i in tqdm(range(M), desc="Precomputing similarity matrix (cslr_subtitle)"):
            cue_text = cues[i]['text']
            for j in range(N):
                if similarity_measure == "cslr_subtitle":
                    seg_sub = sign_segments[j].get('subtitle', '')
                    if seg_sub:
                        sim_matrix[i, j] = 1 if seg_sub == cue_text else -1
                    else:
                        sim_matrix[i, j] = 0
                elif similarity_measure == "cslr_text":
                    seg_text = sign_segments[j].get('text', '')
                    if seg_text:
                        sim_matrix[i, j] = -1
                        seg_texts = seg_text.split('/')
                        for seg_text in seg_texts:
                            if seg_text.lower() in cue_text.lower():
                                probs = sign_segments[j].get('probs', 1)
                                sim_matrix[i, j] = probs
                    else:
                        sim_matrix[i, j] = 0

    elif similarity_measure == "text_embedding":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode sign texts once.
        sign_texts = [(seg.get('text') or "").strip() for seg in sign_segments]
        sign_embeddings = model.encode(sign_texts, show_progress_bar=True)
        
        if tokenize_text_embedding:
            # Initialize an empty similarity matrix.
            sim_matrix = np.zeros((M, N))
            for i, cue in enumerate(cues):
                cue_text = (cue.get('text') or "").strip()
                # Tokenize the cue text into words (customize tokenization as needed)
                tokens = cue_text.split()
                if tokens:
                    # Compute embeddings for each token.
                    token_embeddings = model.encode(tokens, show_progress_bar=False)
                    for j, sign_embedding in enumerate(sign_embeddings):
                        # If the sign text is empty, assign 0 similarity.
                        if not sign_texts[j]:
                            sim_matrix[i, j] = 0
                        else:
                            # Compute the similarity (dot product) between each token and the sign embedding.
                            token_similarities = np.dot(token_embeddings, sign_embedding)
                            # Pool the token similarities based on the text_embedding_pooling method.
                            if text_embedding_pooling == "mean":
                                sim_matrix[i, j] = np.mean(token_similarities)
                            elif text_embedding_pooling == "max":
                                sim_matrix[i, j] = np.max(token_similarities)
                            else:
                                raise ValueError("Invalid text_embedding_pooling value. Use 'mean' or 'max'.")
                else:
                    # If there are no tokens, set similarity to zero.
                    sim_matrix[i, :] = 0
        else:
            # Original behavior: compute embeddings for the full cue texts.
            cue_texts = [cue.get('text') or "" for cue in cues]
            cue_embeddings = model.encode(cue_texts, show_progress_bar=True)
            sim_matrix = np.dot(cue_embeddings, sign_embeddings.T)
            # Zero out columns corresponding to empty sign texts.
            for j, seg in enumerate(sign_segments):
                if not (seg.get('text') or "").strip():
                    sim_matrix[:, j] = 0

    elif similarity_measure == "sign_clip_embedding":
        if tokenize_text_embedding:
            # Ensure we have one tokenized embedding per cue.
            if subtitle_embedding_tokenized is None or len(subtitle_embedding_tokenized) != M:
                raise ValueError(f"Subtitle embedding tokenized mismatch: expected {M} elements, got {len(subtitle_embedding_tokenized) if subtitle_embedding_tokenized is not None else 'None'}")
            # Initialize an empty similarity matrix.
            sim_matrix = np.zeros((M, N))
            for i in range(M):
                token_embeddings = subtitle_embedding_tokenized[i]  # shape: (num_tokens, embedding_dim)
                if token_embeddings.size == 0 or token_embeddings.shape[0] == 0:
                    sim_matrix[i, :] = 0
                else:
                    for j in range(N):
                        sign_embedding = segmentation_embedding[j]  # shape: (embedding_dim,)
                        token_similarities = np.dot(token_embeddings, sign_embedding)
                        if text_embedding_pooling == "mean":
                            sim_matrix[i, j] = np.mean(token_similarities)
                        elif text_embedding_pooling == "max":
                            sim_matrix[i, j] = np.max(token_similarities)
                        else:
                            raise ValueError("Invalid text_embedding_pooling value. Use 'mean' or 'max'.")
        else:
            if subtitle_embedding.shape[0] != M:
                raise ValueError(f"Subtitle embedding mismatch: expected {M} rows, got {subtitle_embedding.shape[0]}")
            if segmentation_embedding.shape[0] != N:
                raise ValueError(f"Segmentation embedding mismatch: expected {N} rows, got {segmentation_embedding.shape[0]}")
            sim_matrix = np.dot(subtitle_embedding, segmentation_embedding.T)
    
    else:
        raise ValueError(f"Unsupported similarity_measure: {similarity_measure}")

    # Compute midpoints once
    cue_midpoints = [(cue['start'] + cue['end']) / 2 for cue in cues]
    sign_midpoints = [(seg['start'] + seg['end']) / 2 for seg in sign_segments]

    normalized_matrix = sim_matrix.copy()
    window_size_row = 50
    window_size_col = int(window_size_row * len(cues) / len(sign_segments))

    if normalization_method == 'sinkhorn':
        row_normalized_matrix = np.zeros_like(sim_matrix)

        for i_start in range(0, M, window_size_col):  # cues block (rows)
            i_end = min(i_start + window_size_col, M)
            cue_block = cues[i_start:i_end]
            cue_block_midpoints = cue_midpoints[i_start:i_end]
            mid_block_cue = np.mean(cue_block_midpoints)

            # Find signs closest to the cue block center
            sign_distances = [abs(sign_mid - mid_block_cue) for sign_mid in sign_midpoints]
            sorted_sign_indices = np.argsort(sign_distances)
            sign_window_indices = sorted_sign_indices[:window_size_row]  # signs block (columns)

            # Extract submatrix for cue block × sign window
            submatrix = sim_matrix[i_start:i_end, :][:, sign_window_indices]
            if submatrix.size == 0:
                continue

            # Apply Sinkhorn
            submatrix = sinkhorn_normalize(submatrix)
            submatrix = submatrix * window_size_col

            # Apply filtering
            submatrix[submatrix <= filter_threshold] = 0.0

            # Write submatrix back into global matrix
            for local_i, global_i in enumerate(range(i_start, i_end)):
                for local_j, global_j in enumerate(sign_window_indices):
                    row_normalized_matrix[global_i, global_j] = submatrix[local_i, local_j]

        normalized_matrix = row_normalized_matrix
    else:
        if normalize_columnwise:
            col_normalized_matrix = np.zeros_like(normalized_matrix)

            for j in range(N):
                sign_mid = sign_midpoints[j]
                cue_distances = [abs(cue_mid - sign_mid) for cue_mid in cue_midpoints]
                sorted_indices = np.argsort(cue_distances)
                window_indices = sorted_indices[:window_size_col]

                values_to_normalize = normalized_matrix[window_indices, j]
                normalized_values = softmax_normalize(values_to_normalize, axis=0, tau=10)
                normalized_values *= window_size_col

                # if j < 20:
                #     print(j, sign_segments[j])
                #     print(values_to_normalize)
                #     print(normalized_values)

                for k, i in enumerate(window_indices):
                    col_normalized_matrix[i, j] = normalized_values[k]

            normalized_matrix = col_normalized_matrix

        if normalize_rowwise:
            row_normalized_matrix = np.zeros_like(normalized_matrix)

            for i in range(M):
                cue_mid = cue_midpoints[i]
                sign_distances = [abs(sign_mid - cue_mid) for sign_mid in sign_midpoints]
                sorted_indices = np.argsort(sign_distances)
                window_indices = sorted_indices[:window_size_row]

                values_to_normalize = normalized_matrix[i, window_indices]

                if normalization_method == 'softmax':
                    normalized_values = softmax_normalize(values_to_normalize, axis=0, tau=10)
                    normalized_values *= window_size_row
                elif normalization_method == 'z-score':
                    normalized_values = zscore_sigmoid_normalize(values_to_normalize, tau=5)
                elif normalization_method == 'sinkhorn':
                    submatrix = normalized_matrix[i:i+1, window_indices]
                    submatrix = sinkhorn_normalize(submatrix)
                    normalized_values = submatrix[0]
                else:
                    raise ValueError(f"Unsupported normalization_method: {normalization_method}")

                # Apply filtering (always)
                normalized_values[normalized_values <= filter_threshold] = 0.0

                for k, j in enumerate(window_indices):
                    row_normalized_matrix[i, j] = normalized_values[k]

            normalized_matrix = row_normalized_matrix

    # print(normalized_matrix)

    return normalized_matrix