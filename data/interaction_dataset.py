import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

import json

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    """Indexes an interaction dataset, then returns frames by index."""
    def __init__(self, dataset_path, max_trajectory_length=None,
            target_type='in_frame', high_res_images=False,
            scene_binary_labels=True):
        self.dataset_path = dataset_path
        self.max_trajectory_length = max_trajectory_length
        self.target_type = target_type
        self.high_res_images = high_res_images # .jpg or .png
        self.scene_binary_labels = scene_binary_labels

        self.object_counts_target_type = 'object_counts_' + self.target_type
        self.frame_extension = '.png' if self.high_res_images else '.jpg'

        # TODO: add option for bounding box object classifications
        self.trajectory_directories = os.listdir(dataset_path)
        self.trajectory_directories.sort()
        if self.max_trajectory_length is None: # Different length trajectories
            self.num_frames = 0
            # List of tuples of (trajectory directory index, frame index)
            self.trajectory_indexes_frames = []
            # TODO: if opening all info files (one per trajectory) is taking
            # too long, can add a master index file in
            # gen/scripts/replay_trajectories that has an array/list of
            # trajectory index to trajectory length (shared tensor, then write
            # to file after all processes are joined)
            for trajectory_i, trajectory_directory in enumerate(
                    self.trajectory_directories):
                info_path = os.path.join(self.dataset_path,
                        trajectory_directory, 'info.json')
                with open(info_path, 'r') as jsonfile:
                    info = json.load(jsonfile)
                self.trajectory_indexes_frames.extend([
                    (trajectory_i, frame_i) for frame_i in range(len(
                        info[self.object_counts_target_type]))])
        else:
            self.num_frames = (len(self.trajectory_directories) *
                    self.max_trajectory_length)

    def __len__(self):
        if self.max_trajectory_length is None:
            return len(self.trajectory_indexes_frames)
        else:
            return self.num_frames

    def __getitem__(self, idx):
        if type(idx) is int or (torch.is_tensor(idx) and len(idx.shape) == 0):
            indexes = [int(idx)]
            singleton = True
        elif torch.is_tensor(idx):
            indexes = idx.tolist()
            singleton = False
        else:
            indexes = list(idx)
            singleton = False

        data = []
        target = []
        for index in indexes:
            if self.max_trajectory_length is None:
                trajectory_directory_index, frame_index = (
                    self.trajectory_indexes_frames[index])
                trajectory_directory = self.trajectory_directories[
                        trajectory_directory_index]
                frame_path = os.path.join(self.dataset_path,
                    trajectory_directory, '%05d' % frame_index +
                    self.frame_extension)
            else:
                trajectory_directory_index = (index //
                        self.max_trajectory_length)
                frame_index = index % self.max_trajectory_length
                trajectory_directory = self.trajectory_directories[
                        trajectory_directory_index]
                frame_path = os.path.join(self.dataset_path,
                        trajectory_directory, '%05d' % frame_index +
                        self.frame_extension)
            # Reshape cv2 BGR to RGB
            img = cv2.imread(frame_path)
            data.append(torch.tensor(cv2.cvtColor(img,
                cv2.COLOR_BGR2RGB)))
            info_path = os.path.join(self.dataset_path,
                    trajectory_directory, 'info.json')
            with open(info_path, 'r') as jsonfile:
                info = json.load(jsonfile)
            raw_target = torch.tensor(info[self.object_counts_target_type][
                frame_index])
            if self.scene_binary_labels:
                raw_target = raw_target > 0
            target.append(raw_target.float())
        if singleton:
            # If input is an int or a tensor with no shape, return sample as an
            # element or lists instead of lists of lists
            data = data[0]
            target = target[0]
        else:
            data = torch.stack(data)
            target = torch.stack(target)

        return data, target

