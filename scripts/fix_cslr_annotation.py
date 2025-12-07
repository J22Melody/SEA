#!/usr/bin/env python3
import os
import argparse
import re
import pandas as pd
from pathlib import Path

# Regex patterns to parse a gloss token.
# This pattern finds tokens like: gloss_text[123.45-678.90]
GLOSS_TOKEN_PATTERN = re.compile(r'(.+?\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\])')
GLOSS_DETAILS_PATTERN = re.compile(r'(.+?)\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)]$')

def parse_gloss_sequence(seq):
    """
    Given a gloss sequence string (e.g. "wait[100-101] dive[101-102]"),
    return a list of dictionaries with keys: text, start, end.
    """
    gloss_tokens = []
    if not isinstance(seq, str) or not seq.strip():
        return gloss_tokens

    # Find all gloss token strings
    tokens = GLOSS_TOKEN_PATTERN.findall(seq)
    for token in tokens:
        m = GLOSS_DETAILS_PATTERN.match(token)
        if m:
            text = m.group(1).strip()
            start = float(m.group(2))
            end = float(m.group(3))
            gloss_tokens.append({"raw": token, "text": text, "start": start, "end": end})
    return gloss_tokens

def assemble_gloss_sequence(gloss_tokens):
    """
    Reassemble gloss tokens back into a string.
    Tokens are formatted as: text[start-end]
    """
    # Here we reconstruct the token (formatting might slightly change)
    tokens = [f"{token['text']}[{token['start']}-{token['end']}]" for token in gloss_tokens]
    return " ".join(tokens)

def compute_overlap(g_start, g_end, sub_start, sub_end):
    """
    Compute the temporal overlap between a gloss (g_start, g_end)
    and a subtitle (sub_start, sub_end).
    """
    return max(0, min(g_end, sub_end) - max(g_start, sub_start))

def process_csv_file(csv_path, out_path):
    """
    Process one CSV file:
      - Sort rows by start_sub.
      - Parse gloss sequences.
      - For each row, check if the last gloss token belongs to the next subtitle.
      - Sort gloss tokens within each row.
      - Save the updated CSV to out_path.
    """
    print(f"Processing file: {csv_path}")
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure start_sub and end_sub are floats
    df['start_sub'] = df['start_sub'].astype(float)
    df['end_sub'] = df['end_sub'].astype(float)
    
    # Sort rows by start_sub
    df.sort_values(by='start_sub', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Parse gloss tokens for each row and store them in a new column
    df['gloss_tokens'] = df['approx gloss sequence'].apply(parse_gloss_sequence)
    
    # Process each row (except the last) to check if the last gloss token belongs to the next subtitle
    for i in range(len(df) - 1):
        current_start = df.at[i, 'start_sub']
        current_end = df.at[i, 'end_sub']
        next_start = df.at[i+1, 'start_sub']
        next_end = df.at[i+1, 'end_sub']
        
        tokens_current = df.at[i, 'gloss_tokens']
        tokens_next = df.at[i+1, 'gloss_tokens']
        
        if tokens_current:
            # Take the last gloss token in current row
            last_token = tokens_current[-1]
            g_start = last_token['start']
            g_end = last_token['end']
            # Compute overlap with current and next subtitles
            overlap_current = compute_overlap(g_start, g_end, current_start, current_end)
            overlap_next = compute_overlap(g_start, g_end, next_start, next_end)
            
            # If the gloss overlaps more with the next subtitle than with the current one,
            # then remove it from the current row and prepend it to the next row.
            if overlap_current < overlap_next:
                print(f"Moving gloss token '{last_token['raw']}' from subtitle {i} to subtitle {i+1}")
                # Remove from current
                tokens_current.pop(-1)
                # Prepend to next row tokens (to preserve order, later we will sort)
                tokens_next.insert(0, last_token)
                # Update the DataFrame columns
                df.at[i, 'gloss_tokens'] = tokens_current
                df.at[i+1, 'gloss_tokens'] = tokens_next

    # For each row, sort gloss tokens by their start time and reassemble the gloss sequence string.
    new_gloss_seqs = []
    for tokens in df['gloss_tokens']:
        # Sort tokens by start time
        tokens_sorted = sorted(tokens, key=lambda token: token['start'])
        new_gloss_seqs.append(assemble_gloss_sequence(tokens_sorted))
    df['approx gloss sequence'] = new_gloss_seqs
    
    # Drop the helper column before saving
    df.drop(columns=['gloss_tokens'], inplace=True)
    
    # Save the fixed CSV to the output path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved fixed CSV to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Fix CSLR CSV annotations")
    parser.add_argument("--cslr_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/continuous_sign_sequences/cslr-raw",
                        help="Directory with CSLR CSV files (searched recursively).")
    parser.add_argument("--cslr_dir_fixed", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/continuous_sign_sequences/cslr-fixed",
                        help="Directory to save fixed CSLR CSV files.")
    args = parser.parse_args()
    
    # Use pathlib to recursively search for CSV files in the input directory.
    input_dir = Path(args.cslr_dir)
    output_dir = Path(args.cslr_dir_fixed)
    
    csv_files = list(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    for csv_file in csv_files:
        # Compute the relative path so we can preserve folder structure if desired.
        rel_path = csv_file.relative_to(input_dir)
        out_file = output_dir / rel_path
        # Ensure the output directory exists
        out_file.parent.mkdir(parents=True, exist_ok=True)
        process_csv_file(csv_file, out_file)

if __name__ == "__main__":
    main()
