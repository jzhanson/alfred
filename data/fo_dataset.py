import json
import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from env.find_one import ACTIONS_DONE

def get_trajectories(paths):
    trajectories = []
    # Iterate through jsons
    for path in paths:
        with open(os.path.join(path, 'traj_data.json')) as jsonfile:
            traj_dict = json.load(jsonfile)
            current_trajectories = []
            current_high_idxs = []
            # Iterate through high_pddls (subgoals) and collect all relevant
            # GotoLocations
            for high_pddl_dict in traj_dict['plan']['high_pddl']:
                # TODO: also add the different types of Find replacements to
                # augment
                if high_pddl_dict['discrete_action']['action'] == \
                        'GotoLocation':
                    current_trajectory = {}
                    current_trajectory['path'] = path
                    # Only one argument for GotoLocation
                    current_trajectory['target'] = \
                            high_pddl_dict['discrete_action']['args'][0]
                    current_trajectory['high_idx'] = high_pddl_dict['high_idx']
                    # Iterate through low_actions and images once later on and
                    # fill these in later
                    current_trajectory['low_actions'] = []
                    current_trajectory['images'] = []
                    current_trajectory['features'] = []
                    current_trajectories.append(current_trajectory)
                    current_high_idxs.append(high_pddl_dict['high_idx'])

            # Iterate through once to get all the low_actions and images
            # corresponding to each GotoLocation and ResNet features.
            #
            # There are 10 more ResNet features (i.e. of shape [N+10, 512, 7,
            # 7]) than there are images, for some reason
            for low_action in traj_dict['plan']['low_actions']:
                if low_action['high_idx'] in current_high_idxs:
                    trajectory_index = \
                            current_high_idxs.index(low_action['high_idx'])
                    current_trajectories[trajectory_index]['low_actions'] \
                            .append(low_action['api_action']['action'])

            # Iterate through images once and grab the images that correspond
            # to a high_idx, one per low action
            seen_low_idxs = []
            trajectory_index_to_last_low_idx = {}
            for i in range(len(traj_dict['images'])):
                image = traj_dict['images'][i]
                # Only save one image per low action, only for the low actions
                # that we care about
                if image['high_idx'] in current_high_idxs and \
                        image['low_idx'] not in seen_low_idxs:
                    trajectory_index = \
                            current_high_idxs.index(image['high_idx'])
                    # Images in traj_data.json are .png where in reality they
                    # are .jpg
                    image_name = image['image_name'].split('.')[0] + '.jpg'
                    current_trajectories[trajectory_index]['images'] \
                            .append(image_name)
                    # Supposedly 10 extra features are at the end and don't
                    # matter --- see models/model/seq2seq_im_mask.py
                    current_trajectories[trajectory_index]['features'] \
                            .append(i)
                    # Also record the high_idx and the low_idx so we can get
                    # the image and feature immediately after the high_pddl
                    # for annotating with the 'Done' action
                    trajectory_index_to_last_low_idx[trajectory_index] = \
                            image['low_idx']
                    seen_low_idxs.append(image['low_idx'])
            # Iterate through one last time and grab the last images and
            # features. Sort by high_idx so we only have to iterate through
            # images once, instead of len(images) *
            # len(high_idx_to_last_low_idx) times.
            trajectory_index_to_last_low_idx = sorted(
                    trajectory_index_to_last_low_idx.items(), key=lambda x:
                    x[0])
            for i in range(len(traj_dict['images'])):
                image = traj_dict['images'][i]
                if image['low_idx'] <= trajectory_index_to_last_low_idx[0][1]:
                    # Keep going until we reach the first low_idx after the
                    # last low_idx in the current high_idx
                    continue
                trajectory_index = trajectory_index_to_last_low_idx[0][0]
                image_name = image['image_name'].split('.')[0] + '.jpg'
                current_trajectories[trajectory_index]['images'].append(
                        image_name)
                current_trajectories[trajectory_index]['features'].append(
                        i)
                current_trajectories[trajectory_index]['low_actions'].append(
                        ACTIONS_DONE)
                trajectory_index_to_last_low_idx.pop(0)
                if len(trajectory_index_to_last_low_idx) == 0:
                    break
            # There may be an edge case where the high_pddl is at the very end
            # so the above loop skips adding an extra frame for the done action
            # because the episode ended. In this case, we choose not to
            # duplicate the last frame since that would be technically
            # incorrect, but print to keep track of if/when this happens
            if len(trajectory_index_to_last_low_idx) > 0:
                print("trajectories not annotated with Done action: " +
                        str(len(trajectory_index_to_last_low_idx)))
            trajectories.extend(current_trajectories)
    return trajectories

def make_fo_dataset():
    splits_path = os.path.join(os.environ['ALFRED_ROOT'], "data/splits/oct21.json")
    task_repo = os.path.join(os.environ['ALFRED_ROOT'], "data/full_2.1.0")

    split_tasks = json.load(open(splits_path))
    #splits = split_tasks.keys()

    training_paths = [os.path.join(task_repo, 'train', item['task']) for item in
            split_tasks['train'][:16]]
    validation_seen_paths = [os.path.join(task_repo, 'valid_seen',
        item['task']) for item in split_tasks['valid_seen'][:16]]
    validation_unseen_paths = [os.path.join(task_repo, 'valid_unseen', \
            item['task']) for item in split_tasks['valid_unseen'][:16]]

    training_trajectories = get_trajectories(training_paths)
    validation_seen_trajectories = get_trajectories(validation_seen_paths)
    validation_unseen_trajectories = get_trajectories(validation_unseen_paths)

    # Save training trajectories as JSON
    if not os.path.isdir(os.path.join(os.environ['ALFRED_ROOT'],
        "data/find_one")):
        os.mkdir(os.path.join(os.environ['ALFRED_ROOT'], "data/find_one"))

    training_outfile_path = os.path.join(os.environ['ALFRED_ROOT'],
            "data/find_one/train.json")
    with open(training_outfile_path, 'w') as outfile:
        json.dump(training_trajectories, outfile, sort_keys=True, indent=4)

    validation_seen_outfile_path = os.path.join(os.environ['ALFRED_ROOT'],
            "data/find_one/valid_seen.json")
    with open(validation_seen_outfile_path, 'w') as outfile:
        json.dump(validation_seen_trajectories, outfile, sort_keys=True,
                indent=4)

    validation_unseen_outfile_path = os.path.join(os.environ['ALFRED_ROOT'],
            "data/find_one/valid_unseen.json")
    with open(validation_unseen_outfile_path, 'w') as outfile:
        json.dump(validation_unseen_trajectories, outfile, sort_keys=True,
                indent=4)

class FindOneDataset(Dataset):
    """
    This class returns trajectories by index.
    """
    def __init__(self, json_path, images=True, features=True,
            transitions=False):
        with open(json_path, 'r') as jsonfile:
            self.trajectories = json.load(jsonfile)
        self.images = images
        self.features = features
        if transitions:
            self.transitions = []
            # To avoid having to iterate through anything, we fill this list
            # with tuples of (trajectory index, transition index within that
            # trajectory)
            self.transitions = []
            for trajectory_index in range(len(self.trajectories)):
                for transition_index in range(len(self.trajectories[
                    trajectory_index]['low_actions'])):
                    self.transitions.append((trajectory_index,
                        transition_index))
        else:
            self.transitions = None

    def __len__(self):
        return len(self.transitions) if self.transitions is not None else \
                len(self.trajectories)

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

        sample = {}
        sample['target'] = []
        sample['low_actions'] = []
        if self.images:
            sample['images'] = []
        if self.features:
            sample['features'] = []

        for index in indexes:
            if self.transitions is not None:
                trajectory_index, transition_index = self.transitions[index]
                sample['target'].append(self.trajectories[trajectory_index][
                    'target'])
                sample['low_actions'].append(self.trajectories[
                    trajectory_index]['low_actions'][transition_index])
                current_path = self.trajectories[trajectory_index]['path']
                # Load the images and/or features of each sample
                if self.images:
                    image_path = os.path.join(current_path, 'raw_images',
                            self.trajectories[trajectory_index]['images'][
                                transition_index])
                    # Reshape cv2 BGR image to RGB
                    sample['images'].append(torch.flip(torch.tensor(cv2
                        .imread(image_path)), [2]))
                if self.features:
                    features = torch.load(os.path.join(current_path,
                        'feat_conv.pt'))
                    sample['features'].append(features[self.trajectories
                            [trajectory_index]['features'][transition_index]])
            else:
                sample['target'].append(self.trajectories[index]['target'])
                sample['low_actions'].append(self.trajectories[index]
                        ['low_actions'])
                current_path = self.trajectories[index]['path']
                # Load the images and/or features of each sample
                if self.images:
                    current_images = []
                    for image_name in self.trajectories[index]['images']:
                        image_path = os.path.join(current_path, 'raw_images',
                                image_name)
                        # Reshape cv2 BGR image to RGB
                        current_images.append(torch.flip(torch.tensor(cv2
                            .imread( image_path)), [2]))
                    sample['images'].append(torch.stack(current_images))
                if self.features:
                    features = torch.load(os.path.join(current_path,
                        'feat_conv.pt'))
                    sample['features'].append(features[self.trajectories
                            [index]['features']])

        if singleton:
            # If input is an int or a tensor with no shape, return sample as an
            # element or lists instead of lists of lists
            sample['target'] = sample['target'][0]
            sample['low_actions'] = sample['low_actions'][0]
            if self.images:
                sample['images'] = sample['images'][0]
            if self.features:
                sample['features'] = sample['features'][0]

        return sample

def collate_trajectories(samples):
    new_samples = {}
    for k in samples[0].keys():
        new_samples[k] = []
        for sample in samples:
            new_samples[k].append(sample[k])
    return new_samples

def get_dataloaders(batch_size=1, transitions=False):
    dataloaders = {}
    for split in ['train', 'valid_seen', 'valid_unseen']:
        fo_dataset = FindOneDataset(os.path.join(
            os.environ['ALFRED_ROOT'], "data/find_one/" + split + ".json"),
            transitions=transitions)
        # Some weird torch 1.1.0 doesn't like me passing collate_fn=None
        if transitions:
            dataloader = DataLoader(fo_dataset, batch_size=batch_size,
                    shuffle=True, num_workers=0)
        else:
            dataloader = DataLoader(fo_dataset, batch_size=batch_size,
                    shuffle=True, num_workers=0, collate_fn=collate_trajectories)
        dataloaders[split] = dataloader
    return dataloaders

if __name__ == '__main__':
    make_fo_dataset()

    dataloaders = get_dataloaders(batch_size=4, transitions=True)
    for sample_batched in dataloaders['train']:
        print(len(sample_batched['low_actions']),
                len(sample_batched['images']), len(sample_batched['features']),
                len(sample_batched['target']))
        break

