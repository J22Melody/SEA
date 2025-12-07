#!/usr/bin/env python3
import os
import pickle
import argparse

def load_data(file_path):
    """Load subtitle data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_video_ids(file_path):
    """Load video IDs from a text file (one per line)."""
    with open(file_path, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]
    return video_ids

def write_vtt_file(episode, subtitles, save_dir):
    """
    Write subtitles for one episode into a VTT file.
    Since start and end are already in VTT format, we use them directly.
    """
    # Sort cues by start time (string sort works as they are zero-padded)
    subtitles.sort(key=lambda cue: cue['start'])
    vtt_path = os.path.join(save_dir, f"{episode}.vtt")
    
    with open(vtt_path, 'w', encoding='utf-8') as f:
        # Write the WEBVTT header
        f.write("WEBVTT\n\n")
        # Write each cue with a cue number
        for i, cue in enumerate(subtitles, start=1):
            start = cue['start']
            end = cue['end']
            text = cue['subtitle']
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    print(f"Written VTT for episode '{episode}' to {vtt_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract subtitles for selected episodes and write as VTT files."
    )
    parser.add_argument("--video_ids", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align.txt",
                        help="Path to text file with video ids (one per line).")
    parser.add_argument("--subtitle_file", type=str,
                        default="/work/youngjoon/new_subtitles.pkl",
                        help="Path to the pickle file containing subtitles.")
    parser.add_argument("--save_dir", type=str,
                        default="/users/zifan/subtitle_align/alternative/aligned_subtitles_youngjoon",
                        help="Directory to store aligned subtitle VTT files.")
    args = parser.parse_args()

    # Verify file existence and prepare the save directory
    if not os.path.exists(args.subtitle_file):
        print(f"Subtitle file not found: {args.subtitle_file}")
        return

    if not os.path.exists(args.video_ids):
        print(f"Video IDs file not found: {args.video_ids}")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    # Load the subtitle data and video IDs list.
    data = load_data(args.subtitle_file)
    video_ids = load_video_ids(args.video_ids)

    # Check if the data dict contains the required keys.
    required_keys = ['episode_name', 'start', 'end', 'duration', 'subtitle']
    if not all(key in data for key in required_keys):
        print("Data does not contain the required keys.")
        return

    # Group subtitles by episode.
    grouped_subtitles = {}
    num_records = len(data['episode_name'])
    for i in range(num_records):
        episode = data['episode_name'][i]
        if episode in video_ids:
            record = {
                'start': data['start'][i],
                'end': data['end'][i],
                'duration': data['duration'][i],
                'subtitle': data['subtitle'][i]
            }
            grouped_subtitles.setdefault(episode, []).append(record)

    # Write each episode's subtitles to a separate VTT file.
    for episode, subtitles in grouped_subtitles.items():
        write_vtt_file(episode, subtitles, args.save_dir)

if __name__ == '__main__':
    main()
