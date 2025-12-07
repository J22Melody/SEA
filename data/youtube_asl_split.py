import os
import random
import argparse

def split_video_ids(curated_ids_file, train_file, val_file, val_size=200, seed=42):
    random.seed(seed)
    
    # Read video IDs from the curated file
    with open(curated_ids_file, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    # Shuffle the video IDs
    random.shuffle(video_ids)
    
    # Split into training and validation sets
    val_ids = video_ids[:val_size]
    train_ids = video_ids[val_size:]
    
    # Write to files
    with open(train_file, "w") as f:
        f.write("\n".join(train_ids))
    
    with open(val_file, "w") as f:
        f.write("\n".join(val_ids))
    
    print(f"Split completed. Training set: {len(train_ids)} IDs, Validation set: {len(val_ids)} IDs.")

def main():
    parser = argparse.ArgumentParser(description="Split curated video IDs into training and validation sets.")
    parser.add_argument("input_dir", nargs="?", default="/scratch/shared/beegfs/zifan/YouTube-ASL", help="Input directory containing curated_ids.txt")
    args = parser.parse_args()
    
    curated_ids_file = os.path.join(args.input_dir, "curated_ids.txt")
    train_file = "data/youtube_asl_train.txt"
    val_file = "data/youtube_asl_val.txt"
    
    if not os.path.exists(curated_ids_file):
        sys.exit(f"ERROR: Curated IDs file {curated_ids_file} does not exist.")
    
    split_video_ids(curated_ids_file, train_file, val_file)

if __name__ == "__main__":
    main()