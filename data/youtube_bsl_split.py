import os
import random

def split_video_ids(videos_path, train_file, val_file, val_size=100, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Get all video files (i.e., all .mp4 files)
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    
    # Extract the video ids by removing the .mp4 suffix
    video_ids = [os.path.splitext(f)[0] for f in video_files]
    
    # Shuffle the video ids randomly
    random.shuffle(video_ids)
    
    # Split the video ids into training and validation sets
    val_ids = video_ids[:val_size]  # First 100 for validation
    train_ids = video_ids[val_size:]  # The rest for training
    
    # Write the video ids to the corresponding files without extra empty line
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_ids))  # Join all IDs with a newline between them
        
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_ids))  # Join all IDs with a newline between them
    
    print(f"Split completed. Training set has {len(train_ids)} ids, Validation set has {len(val_ids)} ids.")

if __name__ == "__main__":
    videos_path = "/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/videos/"
    train_file = "data/youtube_bsl_train.txt"
    val_file = "data/youtube_bsl_val.txt"
    
    # Call the function to split the video ids
    split_video_ids(videos_path, train_file, val_file)
