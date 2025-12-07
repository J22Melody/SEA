import os
import sys
import cv2
import argparse

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def process_videos(input_dir, video_dir=None):
    output_file = os.path.join(input_dir, "curated_fps.txt")
    
    if video_dir:
        videos_dir = os.path.join(input_dir, video_dir)
        if not os.path.exists(videos_dir):
            sys.exit(f"ERROR: Specified video directory {videos_dir} does not exist.")
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
        video_ids = [os.path.splitext(f)[0] for f in video_files]
    else:
        curated_ids_file = os.path.join(input_dir, "curated_ids.txt")
        videos_dir = os.path.join(input_dir, "videos")
        
        if not os.path.exists(curated_ids_file):
            sys.exit(f"ERROR: Curated IDs file {curated_ids_file} does not exist.")
        
        with open(curated_ids_file, "r") as f:
            video_ids = [line.strip() for line in f if line.strip()]
    
    with open(output_file, "w") as f:
        for vid in video_ids:
            video_path = os.path.join(videos_dir, f"{vid}.mp4")
            if not os.path.exists(video_path):
                print(f"Warning: Video file {video_path} does not exist.")
                continue
            
            fps = get_video_fps(video_path)
            if fps is not None:
                f.write(f"{vid} {fps}\n")
                f.flush()  # Ensure data is written incrementally
    
    print(f"FPS extraction completed. Output saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(description="Extract FPS of videos either from curated_ids.txt or directly from a video directory.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/scratch/shared/beegfs/zifan/YouTube-ASL",
        help="Input directory containing curated_ids.txt and videos/"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Optional: Specify a directory relative to input_dir containing videos to extract FPS instead of using curated_ids.txt."
    )
    args = parser.parse_args()
    
    process_videos(args.input_dir, args.video_dir)

if __name__ == "__main__":
    main()
