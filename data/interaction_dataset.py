import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

import json

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    """Indexes an interaction dataset, then returns frames by index."""
    def __init__(self, dataset_path, dataset_type='scene',
            max_trajectory_length=None, high_res_images=False,
            scene_target_type='in_frame', scene_binary_labels=True):
        # TODO: excluded objects, distance threshold
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_trajectory_length = max_trajectory_length
        self.scene_target_type = scene_target_type
        self.high_res_images = high_res_images # .jpg or .png
        self.scene_binary_labels = scene_binary_labels

        self.frame_extension = '.png' if self.high_res_images else '.jpg'

        self.trajectory_directories = os.listdir(dataset_path)
        self.trajectory_directories.sort()
        if self.dataset_type == 'scene':
            self.object_counts_target_type = ('object_counts_' +
                    self.scene_target_type)
            self.target_key = self.object_counts_target_type
        elif self.dataset_type == 'object':
            self.target_key = 'type'

        if (self.dataset_type == 'scene' and self.max_trajectory_length is
                None) or self.dataset_type == 'object':
            # Open all info files if trajectories can be variable size
            # List of tuples of (trajectory directory index, frame index)
            self.trajectory_indexes_frames = []
            # TODO: if opening all info files (one per trajectory) is
            # taking too long, can add a master index file in
            # gen/scripts/replay_trajectories that has an array/list of
            # trajectory index to trajectory length (shared tensor, then
            # write to file after all processes are joined)
            for trajectory_i, trajectory_directory in enumerate(
                    self.trajectory_directories):
                info_path = os.path.join(self.dataset_path,
                        trajectory_directory, self.dataset_type,
                        'info.json')
                with open(info_path, 'r') as jsonfile:
                    info = json.load(jsonfile)
                if self.dataset_type == 'object':
                    # TODO: filter frames by excluded objects, distance
                    pass
                self.trajectory_indexes_frames.extend([
                    (trajectory_i, frame_i) for frame_i in range(len(
                        info[self.target_key]))])
        elif dataset_type == 'scene':
            self.num_frames = (len(self.trajectory_directories) *
                self.max_trajectory_length)

    def __len__(self):
        if self.dataset_type == 'scene':
            if self.max_trajectory_length is None:
                return len(self.trajectory_indexes_frames)
            else:
                return self.num_frames
        elif self.dataset_type == 'object':
                return len(self.trajectory_indexes_frames)

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
            if self.dataset_type == 'scene':
                if self.max_trajectory_length is None:
                    trajectory_directory_index, frame_index = (
                        self.trajectory_indexes_frames[index])
                else:
                    trajectory_directory_index = (index //
                            self.max_trajectory_length)
                    frame_index = index % self.max_trajectory_length
            elif self.dataset_type == 'object':
                trajectory_directory_index, frame_index = (
                    self.trajectory_indexes_frames[index])

            trajectory_directory = self.trajectory_directories[
                    trajectory_directory_index]
            frame_path = os.path.join(self.dataset_path,
                trajectory_directory, self.dataset_type, '%05d' % frame_index +
                self.frame_extension)

            # Reshape cv2 BGR to RGB
            img = cv2.imread(frame_path)
            data.append(torch.tensor(cv2.cvtColor(img,
                cv2.COLOR_BGR2RGB)))
            info_path = os.path.join(self.dataset_path, trajectory_directory,
                    self.dataset_type, 'info.json')
            with open(info_path, 'r') as jsonfile:
                info = json.load(jsonfile)
            raw_target = torch.tensor(info[self.target_key][frame_index])
            if self.dataset_type == 'scene':
                if self.scene_binary_labels:
                    raw_target = raw_target > 0
                raw_target = raw_target.float()
            # TODO: do we need to cast raw_target to anything for object dataset
            target.append(raw_target)

        if singleton:
            # If input is an int or a tensor with no shape, return sample as an
            # element or lists instead of lists of lists
            data = data[0]
            target = target[0]
        else:
            data = torch.stack(data)
            target = torch.stack(target)

        return data, target

