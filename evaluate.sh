# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
# --pred_path_root '/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction' \
# --test_videos_txt 'data/bobsl_align_test.txt' \
# --pr_subs_delta_bias_start 2.7 \
# --pr_subs_delta_bias_end 2.7 \

# Mean and median start offset: 0.21 / 0.21
# Mean and median end offset: 0.67 / 0.66
# Mean and median start offset (abs): 2.01 / 1.26
# Mean and median end offset (abs): 2.20 / 1.42
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 62.41 F1@0.10: 72.78 F1@0.25: 64.10 F1@0.50: 44.61

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
# --pred_path_root '/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/auto_sat_aligned' \
# --test_videos_txt 'data/bobsl_align_test.txt' \

# Mean and median start offset: 0.09 / 0.15
# Mean and median end offset: 0.63 / 0.35
# Mean and median start offset (abs): 1.79 / 0.72
# Mean and median end offset (abs): 1.94 / 0.85
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 71.79 F1@0.10: 74.73 F1@0.25: 67.75 F1@0.50: 54.57

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
# --pred_path_root '/users/zifan/subtitle_align/inference_output/checkpoints_1/subtitle_align_swin/finetune_subtitles/subtitles_postprocessing' \
# --test_videos_txt 'data/bobsl_align_test.txt' \

# Mean and median start offset: -0.39 / -0.06
# Mean and median end offset: 0.17 / 0.27
# Mean and median start offset (abs): 1.62 / 0.78
# Mean and median end offset (abs): 1.70 / 0.88
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 72.33 F1@0.10: 75.23 F1@0.25: 67.61 F1@0.50: 54.49

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
# --pred_path_root '/users/zifan/subtitle_align/inference_output/checkpoints_1/subtitle_align_swin/finetune_subtitles/subtitles_postprocessing' \
# --test_videos_txt 'data/bobsl_align_test_1.txt' \

# Mean and median start offset: -0.19 / -0.15
# Mean and median end offset: 0.54 / 0.29
# Mean and median start offset (abs): 1.49 / 0.76
# Mean and median end offset (abs): 1.54 / 0.91
# Computed over 87847 frames, 462 sentences - Frame-level accuracy: 75.89 F1@0.10: 89.27 F1@0.25: 85.81 F1@0.50: 73.67

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
--pred_path_root '/users/zifan/subtitle_align/alternative/aligned_subtitles' \
--test_videos_txt 'data/bobsl_align_val.txt' \
--pr_subs_delta_bias_start 0.0 \
--pr_subs_delta_bias_end 1.0 \