import os
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import remove_stopwords, get_feature_interval, shift_spottings

from scipy import io
import webvtt
import lmdb
import cv2

import torch

from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, reduce_holistic

from .lmdb_loader import LMDBLoader


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def calc_feat_len(start_time, end_time, fps=25, stride=4):
    len_feats = round(((end_time-start_time)*fps)/float(stride))
    return len_feats

def print_stats(stat, arr):
    print(f'Mean {stat} length samples', np.mean(arr))
    print(f'Standard deviation {stat} length samples', np.std(arr))
    print(f'Min {stat} length samples', np.min(arr))
    print(f'Max {stat} length samples', np.max(arr))
    print(f'80\% percentile {stat} length', np.percentile(arr, 80))
    print(f'90\% percentile {stat} length', np.percentile(arr, 90))
    print(f'95\% percentile {stat} length', np.percentile(arr, 95))

def get_video_frame_count(videos_path, video_name):
    # Ensure the video name has the .mp4 extension
    if not video_name.lower().endswith('.mp4'):
        video_name += '.mp4'
    
    # Complete path to the video file
    video_file = os.path.join(videos_path, video_name)
    
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return None
    else:
        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return total_frames

def load_pose_features(pose_path, stride=1, reduce=False):
    with open(pose_path, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

        if reduce:
            pose_components = ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
            pose_points = {c.name: c.points for c in pose.header.components if c.name in pose_components}

            # Reduced set from the YouTube-ASL paper
            FACE_REDUCED = [0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397, 468, 473]
            POSE_REDUCED = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP']
            pose_points['FACE_LANDMARKS'] = [p for p in pose_points['FACE_LANDMARKS'] if int(p) in FACE_REDUCED]
            pose_points['POSE_LANDMARKS'] = [p for p in pose_points['POSE_LANDMARKS'] if p in POSE_REDUCED]

            pose = pose.get_components(pose_components, pose_points)
        else:
            pose = reduce_holistic(pose)

        pose = pose.normalize()

        feat = np.nan_to_num(pose.body.data)
        # feat = feat[:, :, :, :2]
        feat = feat.reshape(feat.shape[0], -1)
        feat = feat[::stride]

        feat = feat.filled(0)

        return feat, pose.body.fps

class VideoTextDataset(Dataset): 
    
    def __init__(self, mode, opts):
        self.mode = mode
        self.opts = opts

        ### Build data dictionary by iterating through vtt files and captions
        self.data_dict = {}
        self.data_dict["txt"] = []
        self.data_dict["type"] = []
        self.data_dict["pr_fr"] = []
        self.data_dict["pr_to"] = []
        self.data_dict["ep"] = []
        self.data_dict["gt_fr"] = []
        self.data_dict["gt_to"] = []
        self.data_dict["ann_types"] = []
        self.data_dict["ann_times"] = []
        self.data_dict["ann_probs"] = []
        self.data_dict["ann_words"] = []

        ### Load list of train/val/test split videos
        if (self.mode == 'train'): 
            data_paths = open(self.opts.train_videos_txt, "r").read().split('\n')
        elif (self.mode == 'val'): 
            data_paths = open(self.opts.val_videos_txt, "r").read().split('\n')
        elif (self.mode == 'test'):
           data_paths = open(self.opts.test_videos_txt, "r").read().split('\n')
        else: 
            print('Choose mode = "train" or "val" or "test"')
        
        ### For debugging, take a random subset of videos 
        if self.opts.random_subset_data < len(data_paths): 
            random.seed(self.opts.random_subset_data_seed)
            data_paths = random.sample(data_paths, self.opts.random_subset_data)
            print('Number of videos ', len(data_paths), data_paths)

        ### ERROR
        if len(data_paths)==0:
            print('ERROR: Cannot find data')
            return

        ### Loading features
        print('Loading features...')
        self.features = {}
        self.fps = {}
        if self.opts.load_features:
            if self.opts.load_features_from_lmdb:
                lmdb_stride = 2
                lmdb_loader = LMDBLoader(
                    lmdb_path=self.opts.features_path[0],
                    load_type="feats",
                    feat_dim=768,
                    lmdb_stride=lmdb_stride,
                    load_stride=int(self.opts.input_features_stride / lmdb_stride),
                )

                for i, path in tqdm(enumerate(data_paths)):
                    video_name = path
                    end_frame_num = get_video_frame_count(self.opts.videos_path, video_name) - 1

                    episode_features = lmdb_loader.load_sequence(
                        episode_name=video_name,
                        begin_frame=0,
                        end_frame=end_frame_num,
                    )
                    self.features[path] = episode_features

                    if i == 0:
                        print('Example feature for episode:')
                        print(path)
                        print(episode_features.shape)
                        print(episode_features.dtype)
                        print(episode_features[0][:10])
            else:
                for i, path in tqdm(enumerate(data_paths)):
                    feature_list = []  # To store feature arrays from different feature paths
                    fps_list = []      # To store fps values if applicable

                    for base_path in self.opts.features_path:
                        # Determine file extension and flat mode separately for each feature path.
                        is_flat = False
                        if self.opts.load_features_from_pose:
                            ext = '.pose'
                            is_flat = True
                        elif os.path.exists(os.path.join(base_path, data_paths[0]) + '.npy'):
                            ext = '.npy'
                            is_flat = True
                        elif 'features.npy' in os.listdir(os.path.join(base_path, data_paths[0])):
                            ext = '.npy'
                        else:
                            ext = '.mat'
                        
                        # Build the full file path based on whether the features are stored in a flat file.
                        if is_flat:
                            full_path = os.path.join(base_path, data_paths[0]) + ext
                        else:
                            full_path = os.path.join(base_path, path, 'features' + ext)
                        
                        # Load the features if the file exists.
                        if os.path.exists(full_path):
                            if ext == '.pose':
                                features, fps = load_pose_features(
                                    full_path,
                                    stride=self.opts.input_features_stride,
                                    reduce=self.opts.load_features_from_pose_reduce
                                )
                                feature_list.append(features)
                                fps_list.append(fps)
                            elif ext == '.npy':
                                features = np.load(full_path)
                                if 'auto_asvr' in base_path:
                                    features = features[::self.opts.input_features_stride]
                                feature_list.append(features)
                            else:
                                features = io.loadmat(os.path.join(base_path, path, 'features.mat'))['preds']
                                feature_list.append(features)
                        else:
                            print(f"Not found: {full_path}")
                    
                    # Before concatenating, ensure all feature arrays have the same number of rows (dim 0).
                    if feature_list:
                        # Determine the minimum number of rows among all feature arrays.
                        min_rows = min(feat.shape[0] for feat in feature_list)
                        # Trim each feature array to the minimum number of rows.
                        trimmed_features = [feat[:min_rows] for feat in feature_list]
                        # Concatenate the trimmed feature arrays along the second dimension.
                        concatenated_features = np.concatenate(trimmed_features, axis=1)
                        self.features[path] = concatenated_features
                        
                        # Use the first available fps value (adjust as needed).
                        if fps_list:
                            self.fps[path] = fps_list[0]
                    else:
                        print(f"No features found for {path}")

                    # Print example feature info for the first data sample.
                    if i == 0 and path in self.features:
                        print('Example feature for episode:')
                        print(path)
                        print(self.features[path].shape)
                        print(self.features[path].dtype)
                        print(self.features[path][0][:10])

                # is_flat = False
                # if self.opts.load_features_from_pose:
                #     ext='.pose'
                #     is_flat = True
                # elif os.path.exists(os.path.join(self.opts.features_path, data_paths[0]) + '.npy'):
                #     ext='.npy'
                #     is_flat = True
                # elif 'features.npy' in os.listdir(os.path.join(self.opts.features_path, data_paths[0])):
                #     ext='.npy'
                # else:
                #     ext='.mat'
                
                # for i, path in tqdm(enumerate(data_paths)):
                #     if is_flat:
                #         full_path = os.path.join(self.opts.features_path, data_paths[0]) + ext
                #     else:
                #         full_path = os.path.join(self.opts.features_path, path, 'features'+ext)
                #     if os.path.exists(full_path):
                #         if ext=='.pose':
                #             features, fps = load_pose_features(full_path, stride=self.opts.input_features_stride, reduce=self.opts.load_features_from_pose_reduce)
                #             self.features[path] = features
                #             self.fps[path] = fps
                #         elif ext=='.npy':
                #             self.features[path] = np.load(full_path)
                #             # self.features[path] = np.load(full_path)[::self.opts.input_features_stride]
                #             # self.features[path] = np.random.rand(*self.features[path].shape)
                #         else:
                #             self.features[path] = io.loadmat(os.path.join(self.opts.features_path, path, 'features.mat'))['preds']
                #     else:
                #         print(f"Not found: {full_path}")

                #     if i == 0:
                #         print('Example feature for episode:')
                #         print(path)
                #         print(self.features[path].shape)
                #         print(self.features[path].dtype)
                #         print(self.features[path][0][:10])
                    
            vid_episode_keys = self.features.keys()
        else: 
            vid_episode_keys = data_paths

        ### Loading segmentation features
        if self.opts.load_segmentation:
            print('Loading segmentation features...')
            self.segmentation_features = {}
            for path in tqdm(data_paths):
                full_path = os.path.join(self.opts.segmentation_path, data_paths[0]) + '.seg.npy'
                self.segmentation_features[path] = np.load(full_path, allow_pickle=True)[()]

        ### Loading subtitles 
        if self.opts.load_subtitles:
            print('Adding bias to prior subtitle times ', str(self.opts.pr_subs_delta_bias))
            print('Adding bias to GT subtitle times ', str(self.opts.gt_subs_delta_bias))

            print('Loading subtitles associated to sentences...')
            for ep in tqdm(vid_episode_keys):

                # check path for subtitles 
                if os.path.exists(os.path.join(self.opts.pr_sub_path, ep + '/signhd.vtt')):
                    sub_ext_pr = '/signhd.vtt'
                elif os.path.exists(os.path.join(self.opts.pr_sub_path, ep + '.vtt')):
                    sub_ext_pr = '.vtt'
                elif os.path.exists(os.path.join(self.opts.pr_sub_path, ep + '.en.vtt')):
                    sub_ext_pr = '.en.vtt'
                elif os.path.exists(os.path.join(self.opts.pr_sub_path, ep + '.en-GB.vtt')):
                    sub_ext_pr = '.en-GB.vtt'
                elif os.path.exists(os.path.join(self.opts.pr_sub_path, ep + '.srt')):
                    sub_ext_pr = '.srt'
                else:
                    sub_ext_pr = ''
                    print(f"Cannot find subtitle file for: {ep}")
                pr_vtt_path = os.path.join(self.opts.pr_sub_path, ep + sub_ext_pr)

                if self.opts.gt_sub_path:
                    if os.path.exists(os.path.join(self.opts.gt_sub_path, ep + '/signhd.vtt')):
                        sub_ext_gt = '/signhd.vtt'
                    elif os.path.exists(os.path.join(self.opts.gt_sub_path, ep + '.vtt')):
                        sub_ext_gt = '.vtt'
                    elif os.path.exists(os.path.join(self.opts.gt_sub_path, ep + '.en.vtt')):
                        sub_ext_gt = '.en.vtt'
                    elif os.path.exists(os.path.join(self.opts.gt_sub_path, ep + '.en-GB.vtt')):
                        sub_ext_gt = '.en-GB.vtt'
                    elif os.path.exists(os.path.join(self.opts.gt_sub_path, ep + '.srt')):
                        sub_ext_gt = '.srt'
                    else:
                        sub_ext_gt = ''
                        print(f"Cannot find subtitle file for: {ep}")
                    gt_vtt_path = os.path.join(self.opts.gt_sub_path, ep + sub_ext_gt)

                else:
                    gt_vtt_path = pr_vtt_path
                
                pr_subs = webvtt.from_srt(pr_vtt_path) if sub_ext_pr == '.srt' else webvtt.read(pr_vtt_path)
                if self.opts.gt_sub_path:
                    gt_subs = webvtt.from_srt(gt_vtt_path) if sub_ext_gt == '.srt' else webvtt.read(gt_vtt_path)
                    assert len(gt_subs) == len(pr_subs), 'Ground truth subs not the same length as prior subs'    
                else: 
                    gt_subs = pr_subs.copy()

                for idx, pr_sub in enumerate(pr_subs):
                    pr_sub_fr = pr_sub._start + self.opts.pr_subs_delta_bias
                    pr_sub_to = pr_sub._end + self.opts.pr_subs_delta_bias
                    pr_sub_txt = pr_sub.text

                    ### For Train and Val, omit subtitles where GT is not signed and also which are too short or too long
                    ### For Train and Val, omit spottings that are too short
                    remove_sub = 0
                    if self.mode == 'train' or self.mode == 'val': 
                        # remove subs with []
                        if self.opts.gt_sub_path: 
                            if '[' in gt_subs[idx].text: 
                                remove_sub = 1
                        if (pr_sub._end - pr_sub._start < self.opts.min_sent_len_filter) or (pr_sub._end - pr_sub._start > self.opts.max_sent_len_filter):
                            remove_sub = 1
                        if (gt_subs[idx]._end - gt_subs[idx]._start < self.opts.min_sent_len_filter) or (gt_subs[idx]._end - gt_subs[idx]._start > self.opts.max_sent_len_filter):
                            remove_sub = 1
                        sub_text = pr_sub.text
                        if (len(sub_text.split(' ')) < self.opts.min_text_len_filter) or (len(sub_text.split(' ')) > self.opts.max_text_len_filter):
                            remove_sub = 1

                    if not remove_sub:  
                        self.data_dict["ep"].append(ep)
                        self.data_dict["type"].append('subtitle')
                        self.data_dict["pr_fr"].append(pr_sub_fr)
                        self.data_dict["pr_to"].append(pr_sub_to)
                        self.data_dict["txt"].append(pr_sub_txt)

                        if self.opts.gt_sub_path:
                            gt_sub_fr = gt_subs[idx]._start + self.opts.gt_subs_delta_bias
                            gt_sub_to = gt_subs[idx]._end + self.opts.gt_subs_delta_bias
                            self.data_dict["gt_fr"].append(gt_sub_fr)
                            self.data_dict["gt_to"].append(gt_sub_to)
                        else: # TODO is this necessary?
                            self.data_dict["gt_fr"].append(-1)
                            self.data_dict["gt_to"].append(-1)
        
        if self.opts.load_words:
            print('Loading word spottings...') 
            print('Adding bias to word spottings ', str(self.opts.words_delta_bias))
            pad_annot = self.opts.pad_annot
            spottings = pickle.load(open(os.path.join(self.opts.spottings_path), "rb")) 
            for ix, episode in enumerate(spottings['episode_name']): 
                ep = episode.replace('.mp4', '')
                if ep in data_paths and spottings['annot_prob'][ix] > self.opts.conf_thresh_annot and spottings['annot_type'][ix] in ["M*", "D*"]: # (spottings['annot_type'][ix] in ["A", "E", "N"] or spottings['annot_prob'][ix] > self.opts.conf_thresh_annot):
                    time = spottings['annot_time'][ix]
                    if self.opts.shift_spottings: 
                        time = shift_spottings([spottings['annot_type'][ix]], [time], fps=self.opts.fps)      
                        time = time[0]

                    time += self.opts.words_delta_bias 
                    self.data_dict["ep"].append(ep)
                    self.data_dict["type"].append([spottings['annot_type'][ix]]) 
                    self.data_dict["txt"].append(spottings['annot_word'][ix])
                    
                    ### Prior and GT are the same for word alignment
                    ### We jitter in get_item function
                    self.data_dict["pr_fr"].append(time-pad_annot)
                    self.data_dict["pr_to"].append(time+pad_annot)
                    self.data_dict["gt_fr"].append(time-pad_annot)
                    self.data_dict["gt_to"].append(time+pad_annot)

        self.max_feat_len, self.max_text_len = self.calc_stats()

        if self.opts.max_feat_len > 0:
            self.max_feat_len = self.opts.max_feat_len
        if self.opts.max_text_len > 0:
            self.max_text_len = self.opts.max_text_len

        self.shuffled_indices = self.shuffle()

    def calc_stats(self):
        n_samps = len(self.data_dict["txt"])
        print('mode ', self.mode)
        print('number of samples', n_samps)
        all_feat_lens = [
            calc_feat_len(self.data_dict['pr_fr'][ii], self.data_dict['pr_to'][ii], self.opts.fps if self.opts.fps != -1 else self.fps[self.data_dict["ep"][ii]], self.opts.input_features_stride)
            for ii in range(n_samps)
        ]
        print_stats("feats", all_feat_lens)
        all_text_lens = [
                len(self.data_dict['txt'][ii]) for ii in range(n_samps)
            ]
        print_stats("texts", all_text_lens)
        max_feat_len = np.max(all_feat_lens)
        max_text_len = np.max(all_text_lens)
        return max_feat_len, max_text_len
            
    def __len__(self): 
        return len(self.data_dict["txt"])

    def shuffle(self):
        list_ix = list(np.arange(len(self.data_dict["txt"])))
        random.shuffle(list_ix)
        return list_ix

    def __getitem__(self, index):

        if self.opts.shuffle_getitem: 
            index = self.shuffled_indices[index]

        out_dict = {}

        ### Load all information
        text = self.data_dict["txt"][index]
        ep = self.data_dict["ep"][index]
        pr_fr = self.data_dict["pr_fr"][index]
        pr_to = self.data_dict["pr_to"][index]
        gt_fr = self.data_dict["gt_fr"][index]
        gt_to = self.data_dict["gt_to"][index]
        if self.opts.load_features: 
            ep_feats = self.features[ep]
        if self.opts.load_segmentation: 
            seg_feats = self.segmentation_features[ep]

        # if text == 'round': 
        #     print(text, ep, gt_fr, gt_to, ep_feats.shape ep_feats[12285:12291,15])

        # pr_fr and pr_to are the prior start and end times (not frames)
        # wind_fr and wind_to are the window start and end times (not frames)
        # in the fixed_feat_len version (note this is also in seconds not frames), the window size is fixed
        # in the fixed_feat_len=0 version the start and end of the window are just pr_fr and pr_to

        ### Jitter prior 
        if self.opts.jitter_towards_gt:
            pr_fr, pr_to = self.jitter_towards_gt(pr_fr, pr_to, gt_fr, gt_to)
        if self.opts.jitter_mirror_gt:
            pr_fr, pr_to = self.jitter_mirror_gt(pr_fr, pr_to, gt_fr, gt_to)
        if self.opts.jitter_location:
            pr_fr, pr_to = self.jitter_pr_fr_to(pr_fr, pr_to)
        if self.opts.jitter_width_secs>0:
            pr_fr, pr_to = self.jitter_width(pr_fr, pr_to)

        fps = self.fps[ep] if self.opts.fps == -1 else self.opts.fps

        ### Set length of prior window
        ### Pad window to fixed_feat_len width and shift prior within window
        if self.opts.fixed_feat_len > 0:
            wind_fr, wind_to = self.pad_window(pr_fr, pr_to, ep_feats, fps)
        else:
            wind_fr, wind_to = pr_fr, pr_to
            
        ### Retrieve matching window and augment text
        out_dict["orig_txt"] = text
        out_dict["txt"] = self.augment_text(self.clean_text(text.split(' ')[0:self.max_text_len]))

        if self.opts.load_features: 
            if not self.opts.pool_feats:
                out_dict["feats"] = self.return_samp_feats(ep_feats, wind_fr, wind_to, fps=fps).astype(np.single)
            else: 
                out_dict["feats"] = self.pool_feats(ep_feats, wind_fr, wind_to).astype(np.single)
            out_dict["feats_mask"] = (np.sum(out_dict["feats"],1)==0)*1
            if np.max(out_dict["feats_mask"]) > 0:
                out_dict["feats_len"] = np.where((np.sum(out_dict["feats"],1)==0)*1==1)[0][0]
            else:
                out_dict["feats_len"]=0
        
        if self.opts.load_segmentation:
            out_dict["seg_sign_feats"] = self.return_seg_feats(seg_feats['sign'], wind_fr, wind_to).astype(np.single)
            out_dict["seg_sent_feats"] = self.return_seg_feats(seg_feats['sentence'], wind_fr, wind_to).astype(np.single)

        out_dict["pr_fr_to"] = np.array([pr_fr, pr_to]).astype(np.single)  
        out_dict["gt_fr_to"] = np.array([gt_fr, gt_to]).astype(np.single)  
        out_dict["wind_fr_to"] = np.array([wind_fr, wind_to]).astype(np.single)  

        if self.opts.load_features: 
            out_dict['pr_vec'] = self.times_to_labels_vec(out_dict["pr_fr_to"], out_dict["wind_fr_to"], out_dict["feats"]).astype(np.single)
            out_dict['gt_vec'] = self.times_to_labels_vec(out_dict["gt_fr_to"], out_dict["wind_fr_to"], out_dict["feats"]).astype(np.single)  
        out_dict['path'] = ep

        # if self.opts.debug:
        #     print(out_dict["orig_txt"])
        #     print(out_dict["feats"].shape)
        #     print(out_dict['pr_vec'].shape)
        #     print(np.where(out_dict['pr_vec'][:, 0] == 1))
        #     print(out_dict['gt_vec'].shape)
        #     print(np.where(out_dict['gt_vec'][:, 0] == 1))

        return out_dict

    def jitter_towards_gt(self, pr_fr, pr_to, gt_fr, gt_to):
        # jitter fr and to separately
        # pr_fr = random.uniform(pr_fr, gt_fr)
        # pr_to = random.uniform(pr_to, gt_to)

        # jitter fr and to together
        shift_ratio = random.uniform(0, 1)
        pr_fr = pr_fr + shift_ratio * (gt_fr - pr_fr)
        pr_to = pr_to + shift_ratio * (gt_to - pr_to)

        return pr_fr, pr_to 

    def jitter_mirror_gt(self, pr_fr, pr_to, gt_fr, gt_to, probability=0.5):
        if random.uniform(0, 1) > probability:
            pr_fr = pr_fr + 2 * (gt_fr - pr_fr)
            pr_to = pr_to + 2 * (gt_to - pr_to)
        return pr_fr, pr_to 

    def jitter_pr_fr_to(self, pr_fr, pr_to):
        if self.opts.jitter_abs: 
            shift = random.uniform(-self.opts.jitter_loc_quantity,self.opts.jitter_loc_quantity)
        else: 
            shift = random.uniform(-self.opts.jitter_loc_quantity,self.opts.jitter_loc_quantity)*(pr_to - pr_fr)

        pr_fr = pr_fr + shift
        pr_to = pr_to + shift
        return pr_fr, pr_to 
    
    def jitter_width(self, pr_fr, pr_to):
        centre = (pr_to + pr_fr)/2
        old_width = pr_to-centre
        new_width = random.uniform(max(0.25, old_width-self.opts.jitter_width_secs), old_width+self.opts.jitter_width_secs)
        pr_to = centre + new_width
        pr_fr = centre - new_width
        return pr_fr, pr_to

    def pad_window(self, pr_fr, pr_to, ep_feats, fps):
        # random padding around ground truth or prior 
        window_centre = (pr_to+pr_fr)/2
        if random.random() > self.opts.negatives_percent:
            if self.opts.centre_window: # no shift
                shift = 0
            else:
                shift = random.uniform(-0.5*self.opts.fixed_feat_len, 0.5*self.opts.fixed_feat_len)
        else: # if negative then choose a random starting point just before or just after the prior window 
            if random.random() > 0.5:
                shift = random.uniform(pr_fr - self.opts.fixed_feat_len, pr_fr)
            else: 
                shift = random.uniform(pr_to, pr_to + self.opts.fixed_feat_len)

        wind_fr = window_centre - 0.5*self.opts.fixed_feat_len + shift
        wind_to = window_centre + 0.5*self.opts.fixed_feat_len + shift
    
        ### ensure fits in video
        wind_fr = np.clip(wind_fr,0,len(ep_feats)*self.opts.input_features_stride/fps-self.opts.fixed_feat_len)
        wind_to = wind_fr + self.opts.fixed_feat_len
        wind_to = np.clip(wind_to,0,len(ep_feats)*self.opts.input_features_stride/fps)
        wind_fr = wind_to - self.opts.fixed_feat_len
        return wind_fr, wind_to

    ### Index into episode for features, subsample, augment, and add padding
    def return_samp_feats(self, ep_feats, wind_fr, wind_to, fps):
        t0_ix, t1_ix, feats = get_feature_interval(
                            ep_feats,
                            t0_sec=wind_fr,
                            t1_sec=wind_to,
                            fps=fps,
                            clip_stride=self.opts.input_features_stride,
                        )
        if self.opts.subsample_stride > 1:
            # TODO: check this 
            print("WARNING check implementation of stride > 1")
            feats = feats[0:len(feats):self.opts.subsample_stride]
        if self.mode == "train":
            feats = self.augment_feats(feats)

        # if self.opts.debug:
        #     print(fps)
        #     print(wind_fr)
        #     print(wind_to)
        #     print(ep_feats.shape)

        #     print(t0_ix)
        #     print(t1_ix)
        #     print(feats.shape)

        ### fixed window or pad to maximum feature length?
        ## note that wind_len is in frames not seconds
        if self.opts.fixed_feat_len>0:
            wind_len = round(self.opts.fixed_feat_len*fps/self.opts.input_features_stride)
        else:
            wind_len = self.max_feat_len
        
        npad = wind_len - feats.shape[0]
        # TODO debug check 
        # if self.opts.fixed_feat_len: 
        #     assert npad == 0
        if npad < 0 :
            # print(f"WARNING - padding should not be less than 0. npad, wind len, feats, fps = ", npad, wind_len, feats.shape[0], fps)
            # import pdb; pdb.set_trace()
            npad = 0
            feats = feats[:wind_len, :]
        if self.opts.pad_start_features:
            feats = np.pad(feats, [(int(npad), 0), (0, 0)])
        else:
            feats = np.pad(feats, [(0, int(npad)), (0, 0)])
        return feats

    ### Index into episode for segmentation features
    def return_seg_feats(self, ep_feats, wind_fr, wind_to):
        _, _, feats = get_feature_interval(
                            ep_feats.squeeze(),
                            t0_sec=wind_fr,
                            t1_sec=wind_to,
                            clip_stride=1,
                        )

        wind_len = wind_len * self.opts.input_features_stride
        npad = wind_len - feats.shape[0]
        # TODO debug check 
        # if self.opts.fixed_feat_len: 
        #     assert npad == 0
        if npad < 0 :
            # print(f"WARNING - padding should not be less than 0. npad, wind len, feats = ", npad, wind_len, feats.shape[0])
            # import pdb; pdb.set_trace()
            npad = 0
            feats = feats[:wind_len, :]
        if self.opts.pad_start_features:
            feats = np.pad(feats, [(int(npad), 0), (0, 0)])
        else:
            feats = np.pad(feats, [(0, int(npad)), (0, 0)])
        return feats

    def pool_feats(self, ep_feats, wind_fr, wind_to):
        _, _, feats = get_feature_interval(
                    ep_feats,
                    t0_sec=wind_fr,
                    t1_sec=wind_to,
                    clip_stride=self.opts.input_features_stride,
                )
        if self.opts.subsample_stride > 1:
            # TODO: check this 
            print("WARNING check implementation of stride > 1")
            feats = feats[0:len(feats):self.opts.subsample_stride]
        if self.mode == "train":
            feats = self.augment_feats(feats)
        feats = np.expand_dims(np.mean(feats, axis=0), axis=0)
        return feats

    ### Feature augmentation: shuffling and drop
    def augment_feats(self, feats):
        augmented_feats = []
        if self.opts.drop_feats > 0:
            for w in feats:
                # with 85% probability keep the feature
                if random.random() > self.opts.drop_feats:
                    augmented_feats.append(w)
        if self.opts.shuffle_feats > 0:
            if random.random() < self.opts.shuffle_feats:
                random.shuffle(augmented_feats)
        if len(augmented_feats) == 0:
            augmented_feats = feats
        return np.asarray(augmented_feats)
        
    ### Word augmentation: shuffling and drop
    def augment_text(self, words):
        augmented_words = []
        if self.opts.drop_words_subs > 0:
            for w in words:
                # with 85% probability keep the word
                if random.random() > self.opts.drop_words_subs:
                    augmented_words.append(w)
        if self.opts.shuffle_words_subs > 0:
            if random.random() < self.opts.shuffle_words_subs:
                random.shuffle(augmented_words)
        if len(augmented_words) == 0:
            augmented_words = words
        text = ' '.join(augmented_words)
        return text

    def clean_text(self,arr):
        if self.opts.remove_stopwords: 
            arr = [w for w in arr if w not in stop_words]
        ## remove possessifs
        arr = [w.replace("'s", "").replace("'","") for w in arr]
        if self.opts.lemmatize_words:
            # TODO: change this lemmatization function
            try: 
                arr = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in arr]
            except:
                arr = arr
        if self.opts.stem_words:
            arr = [stemmer.stem(w) for w in arr]
        return arr

    def times_to_labels_vec(self, start_end_sample, start_end_window, samp_feats): 
        start_end_window_len = start_end_window[1] - start_end_window[0]
        start_frame = np.round(((start_end_sample[0] - start_end_window[0])/start_end_window_len)*len(samp_feats))
        end_frame = np.round(((start_end_sample[1] - start_end_window[0])/start_end_window_len)*len(samp_feats))
        start_frame = int(np.clip(start_frame,0,len(samp_feats)))
        end_frame = int(np.clip(end_frame,0,len(samp_feats)))
        if start_frame == end_frame:
            end_frame += 1

        vector = np.zeros((len(samp_feats),1), dtype='float32')
        vector[start_frame:end_frame, :] = 1
        return vector

    def pad_tokens(self, tokens):
        padding = self.max_tok_len - len(tokens)
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_tok_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        return tokens, mask
