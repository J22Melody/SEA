import subprocess
import os
import csv
import argparse
from tqdm import tqdm

def get_fps(file_path):
    """
    Uses ffprobe to extract the frame rate (FPS) of the given video file.
    Returns the FPS as a float, or None if not found.
    """
    cmd = [
        'ffprobe',
        '-v', '0',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        num, denom = output.split('/')
        fps = float(num) / float(denom)
        return round(fps, 3)
    except Exception as e:
        print(f"Error getting FPS for {file_path}: {e}")
        return None
        
def save_csv(data, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'fps'])
        writer.writeheader()
        writer.writerows(data)

def summarize_fps(directory, output_csv='fps_summary.csv', save_every=100):
    summary = []
    mp4_files = [f for f in os.listdir(directory) if f.lower().endswith('.mp4')]

    # 👇 Ensure output is saved to script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_csv)

    for idx, fname in enumerate(tqdm(mp4_files, desc='Processing videos')):
        fpath = os.path.join(directory, fname)
        fps = get_fps(fpath)
        summary.append({'filename': fname, 'fps': fps})

        if (idx + 1) % save_every == 0:
            save_csv(summary, output_path)
            print(f"Progress saved after {idx + 1} videos to {output_path}")

    save_csv(summary, output_path)
    print(f"\nFinal FPS summary written to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize FPS of .mp4 files in a directory.')
    parser.add_argument('directory', help='Directory containing MP4 files')
    parser.add_argument('--output', default='fps_summary.csv', help='Output CSV filename')
    parser.add_argument('--save-every', type=int, default=100, help='Save progress every N videos')
    args = parser.parse_args()

    summarize_fps(args.directory, args.output, args.save_every)
