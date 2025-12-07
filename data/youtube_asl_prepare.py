import os
import sys
import json
import argparse
from collections import defaultdict

def convert_json_to_vtt(json_path, vtt_path):
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        
        if "events" not in data:
            print(f"Skipping {json_path}, invalid format.")
            return False
        
        with open(vtt_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for event in data["events"]:
                start_time = event["tStartMs"]
                duration = event["dDurationMs"]
                end_time = start_time + duration
                
                start = format_time(start_time)
                end = format_time(end_time)
                text = " ".join(seg["utf8"] for seg in event.get("segs", []) if "utf8" in seg)
                
                vtt_file.write(f"{start} --> {end}\n{text}\n\n")
        return True
    except Exception as e:
        print(f"Error converting {json_path} to VTT: {e}")
        return False

def format_time(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace(".", ",")

def curate_youtube_asl(input_dir):
    metadata_file = os.path.join(input_dir, "youtube-asl_youtube_asl_video_ids.txt")
    downloads_dir = os.path.join(input_dir, "downloads")
    videos_dir = os.path.join(input_dir, "videos")
    mediapipe_dir = os.path.join(input_dir, "mediapipe")
    subtitles_dir = os.path.join(input_dir, "subtitles")
    curated_ids_file = os.path.join(input_dir, "curated_ids.txt")
    
    if not os.path.exists(metadata_file):
        sys.exit(f"ERROR: Metadata file {metadata_file} does not exist.")
    
    with open(metadata_file, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    priority_json_files = [
        "English en.json", "en.json", "en-US.json", "American Sign Language ase.json"
    ]
    additional_json_patterns = ["en-US.json", "en-CA.json", "en-GB.json"]
    
    found_count_selected_json = 0
    converted_count = 0
    curated_ids = []
    
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(mediapipe_dir, exist_ok=True)
    os.makedirs(subtitles_dir, exist_ok=True)
    
    for vid in video_ids:
        video_download_dir = os.path.join(downloads_dir, vid)
        
        if not os.path.exists(video_download_dir):
            print(f"Video {vid} directory does not exist.")
            continue
        
        mp4_files = [f for f in os.listdir(video_download_dir) if f.endswith('.mp4')]
        pose_files = [f for f in os.listdir(video_download_dir) if f.endswith('.pose')]
        json_files = [f for f in os.listdir(video_download_dir) if f.endswith('.json')]
        
        if len(mp4_files) == 1 and len(pose_files) == 1 and json_files:
            selected_json = None
            for priority in priority_json_files:
                if priority in json_files:
                    selected_json = priority
                    break
            
            if not selected_json:
                for json_file in json_files:
                    if any(pattern in json_file for pattern in additional_json_patterns):
                        selected_json = json_file
                        break
            
            if selected_json:
                found_count_selected_json += 1
                mp4_src = os.path.join(video_download_dir, mp4_files[0])
                pose_src = os.path.join(video_download_dir, pose_files[0])
                json_src = os.path.join(video_download_dir, selected_json)
                vtt_dst = os.path.join(subtitles_dir, f"{vid}.vtt")
                
                mp4_dst = os.path.join(videos_dir, f"{vid}.mp4")
                pose_dst = os.path.join(mediapipe_dir, f"{vid}.pose")
                
                if os.path.exists(mp4_dst) or os.path.islink(mp4_dst):
                    os.remove(mp4_dst)
                os.symlink(mp4_src, mp4_dst)
                
                if os.path.exists(pose_dst) or os.path.islink(pose_dst):
                    os.remove(pose_dst)
                os.symlink(pose_src, pose_dst)
                
                if os.path.exists(vtt_dst):
                    os.remove(vtt_dst)
                if convert_json_to_vtt(json_src, vtt_dst):
                    converted_count += 1
                    curated_ids.append(vid)
    
    with open(curated_ids_file, "w") as f:
        for vid in curated_ids:
            f.write(f"{vid}\n")
    
    print(f"Step 1: Found {found_count_selected_json} videos with all required files and a selected JSON file out of {len(video_ids)} total IDs.")
    print(f"Step 2: Symbolic links created for videos and pose files.")
    print(f"Step 3: Subtitles converted to VTT format. Successfully converted {converted_count} files.")
    print(f"Step 4: Curated IDs saved to {curated_ids_file}.")

def main():
    parser = argparse.ArgumentParser(description="Curate the YouTube-ASL video subtitle dataset.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/scratch/shared/beegfs/zifan/YouTube-ASL",
        help="Input directory for the YouTube-ASL dataset (default: %(default)s)"
    )
    args = parser.parse_args()
    curate_youtube_asl(args.input_dir)

if __name__ == "__main__":
    main()