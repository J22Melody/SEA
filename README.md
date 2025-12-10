# Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing

## Paper Link
- [arXiv preprint](https://arxiv.org/abs/2512.08094)
- Authors: Zifan Jiang, Youngjoon Jang, Liliane Momeni, Gül Varol, Sarah Ebling, Andrew Zisserman

<img width="1318" height="507" alt="image" src="https://github.com/user-attachments/assets/795e8846-45bd-469c-80c8-1282991a1d38" />

## Environment
- Python `3.12`
- Recommended manager: `conda`
- Setup:
  ```bash
  conda env create -f environment.yml
  ```

## Run SEA (on BOBSL validation set for example)

### 0. Pose Estimation

> See more on https://github.com/sign-language-processing/pose.

- Install [pose-format](https://arxiv.org/abs/2310.09066) to run MediaPipe Holistic pose estimation from videos:
  ```
  pip install pose-format
  ```
- Input: `*.mp4`
- Output: `*.pose` (binary pose format)
- Example:
  ```bash
  videos_to_poses --num-workers 32 --format mediapipe --directory ~/BOBSL/derivatives/original_videos --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true" 
  ```

### 1. Segment

> See more on https://github.com/J22Melody/segmentation/tree/bsl.

- Install the [linguistic segmenter](https://aclanthology.org/2023.findings-emnlp.846/) to segment signs based on poses:
  ```
  pip install "git+https://github.com/J22Melody/segmentation@bsl"
  ```
- Input: `*.pose`
- Output: `*.eaf` (ELAN files for segmentation: inspection and annotation)
- Example:
  ```bash
  python segmentation.py --sign-b-threshold 30 --sign-o-threshold 50 --num_workers 4  --video_ids ./data/bobsl_align_val.txt --pose_dir ~/BOBSL/derivatives/video_features/mediapipe_v2_refine_face_complexity_2 --save_dir ~/BOBSL/sea_demo/segmentation
  ```

### 2. Embed (Optional)

> This step requires one GPU, the following tested on a NVIDIA Tesla P40 24GB. See more on https://github.com/J22Melody/fairseq/tree/main/examples/MMPT#inference.

- Install [SignCLIP](https://aclanthology.org/2024.emnlp-main.518/) to embed text and signs (model weigths have to be downloaded):
  ```
  git clone git@github.com:J22Melody/fairseq.git
  cd fairseq
  conda env update -n sea -f environment_inference.yml
  cd examples/MMPT
  pip install .
  ```
- Input: `*.pose, *.vtt`
- Output: `*.npy` (embeddings for signing and subtitle units)
- Example to embed signs:
  ```bash
  python scripts_bsl/extract_episode_features.py --video_ids ~/SEA/data/bobsl_align_val.txt --mode=segmentation --model_name bsl --language_tag "<en> <bfi>" --batch_size=32 --segmentation_dir ~/BOBSL/sea_demo/segmentation/E4s-1_30_50 --save_dir ~/BOBSL/sea_demo/segmentation_embedding/E4s-1_30_50/sign_clip 
  ```
- Example to embed subtitles:
  ```bash
  python scripts_bsl/extract_episode_features.py --video_ids ~/SEA/data/bobsl_align_val.txt --mode=subtitle --model_name bsl --language_tag "<en> <bfi>" --batch_size=1024 --subtitle_dir ~/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction --save_dir ~/BOBSL/sea_demo/subtitle_embedding/sign_clip
  ```

### 3.a. Align without Embeddings (Segment and Align)
- Input: `*.eaf, *.vtt`
- Output: `*.updated.eaf, *.vtt` (ELAN files after alignment and aligned subtitle files)
- Example:
  ```bash
  python align.py --overwrite --mode=inference --video_ids ./data/bobsl_align_val.txt  --num_workers=4 --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 --dp_max_gap 10 --dp_window_size 50 --sign-b-threshold 30 --sign-o-threshold 50 --pr_subs_delta_bias_start 2.6  --pr_subs_delta_bias_end 2.1 --similarity_measure none --segmentation_dir ~/BOBSL/sea_demo/segmentation --save_dir ~/BOBSL/sea_demo/aligned_subtitles
  ```
- Output metrics (if ground truth is provided):
  ```bash
    Metric                         | Result
    -------------------------------+------------
    Total frames                   | 245614
    Total sentences                | 1973
    Mean/median start offset       | -0.50/-0.26
    Mean/median end offset         | -1.04/-0.88
    Mean/median start offset (abs) | 0.93/0.54
    Mean/median end offset (abs)   | 1.29/0.99
    Frame-level accuracy           | 80.68
    F1@0.10                        | 83.07
    F1@0.25                        | 79.32
    F1@0.50                        | 66.24
  ```

### 3.b. Align with Embeddings (Segment, Embed, and Align)
- Input: `*.eaf, *.vtt, *.npy`
- Output: `*.updated.eaf, *.vtt` (ELAN files after alignment and aligned subtitle files)
- Example:
  ```bash
  python align.py --overwrite --mode=inference --video_ids ./data/bobsl_align_val.txt  --num_workers=4 --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 --dp_max_gap 10 --dp_window_size 50 --sign-b-threshold 30 --sign-o-threshold 50 --pr_subs_delta_bias_start 2.6  --pr_subs_delta_bias_end 2.1 --similarity_measure sign_clip_embedding --similarity_weight 10  --segmentation_dir ~/BOBSL/sea_demo/segmentation --subtitle_embedding_dir ~/BOBSL/sea_demo/subtitle_embedding/sign_clip/ --segmentation_embedding_dir ~/BOBSL/sea_demo/segmentation_embedding/E4s-1_30_50/sign_clip --save_dir ~/BOBSL/sea_demo/aligned_subtitles
  ```
- Output metrics (if ground truth is provided):
  ```bash
    Metric                         | Result
    -------------------------------+------------
    Total frames                   | 245614
    Total sentences                | 1973
    Mean/median start offset       | -0.36/-0.18
    Mean/median end offset         | -0.91/-0.77
    Mean/median start offset (abs) | 0.80/0.40
    Mean/median end offset (abs)   | 1.16/0.87
    Frame-level accuracy           | 82.52
    F1@0.10                        | 86.37
    F1@0.25                        | 82.92
    F1@0.50                        | 72.23
  ```

### File Output Structure

You should expect the following output files, including intermediate segmentation and embedding as well as final alignment results.

```bash
/users/zifan/BOBSL/sea_demo/
├── aligned_subtitles
│   ├── 5224144816887051284.vtt
│   ├── 5242317681679687839.vtt
│   ├── 5294309549287947552.vtt
│   └── 5439409006429129628.vtt
├── segmentation
│   └── E4s-1_30_50
│       ├── 5224144816887051284.eaf
│       ├── 5224144816887051284_updated.eaf
│       ├── 5242317681679687839.eaf
│       ├── 5242317681679687839_updated.eaf
│       ├── 5294309549287947552.eaf
│       ├── 5294309549287947552_updated.eaf
│       ├── 5439409006429129628.eaf
│       └── 5439409006429129628_updated.eaf
├── segmentation_embedding
│   └── E4s-1_30_50
│       └── sign_clip
│           ├── 5224144816887051284.npy
│           ├── 5242317681679687839.npy
│           ├── 5294309549287947552.npy
│           └── 5439409006429129628.npy
└── subtitle_embedding
    └── sign_clip
        ├── 5224144816887051284.npy
        ├── 5242317681679687839.npy
        ├── 5294309549287947552.npy
        └── 5439409006429129628.npy
```

### ELAN Visualization

<img width="1718" height="606" alt="image" src="https://github.com/user-attachments/assets/a54c14f1-1981-45d5-8af3-6d37ed61f4f8" />

## Citation
```
@article{jiang2025segment,
  title   = {Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing},
  author  = {Jiang, Zifan and Jang, Youngjoon and Momeni, Liliane and Varol, G{\"u}l and Ebling, Sarah and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:2512.08094},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.08094}
}
```
