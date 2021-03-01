from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from torchvision import transforms

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.future import graph

from functools import reduce

class SingleLayerCNN(nn.Module):
    def __init__(self, input_width=300, input_height=300, output_size=64):
        super(SingleLayerCNN, self).__init__()

    def forward(self, frame):
        pass


class LSTMPolicy(nn.Module):
    def __init__(self, num_actions=12, visual_feature_size=512,
            superpixels_feature_size=512, prev_action_size=16,
            lstm_hidden_size=512, dropout=0, action_fc_units=[],
            visual_fc_units=[], prev_action_after_lstm=False):

        super(LSTMPolicy, self).__init__()

        self.num_actions = num_actions
        self.visual_feature_size = visual_feature_size
        self.superpixels_feature_size = superpixels_feature_size
        self.prev_action_size = prev_action_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = dropout
        self.action_fc_units = action_fc_units
        self.visual_fc_units = visual_fc_units
        self.prev_action_after_lstm = prev_action_after_lstm

        lstm_input_size = self.visual_feature_size + self.prev_action_size

        # Could use nn.LSTM but we probably will never run/train the policy on
        # actual sequences
        # No dropout applied to single layer LSTM
        self.lstm = nn.LSTMCell(input_size=lstm_input_size,
                hidden_size=self.lstm_hidden_size,
                bias=True)

        if self.prev_action_after_lstm:
            fc_input_size = self.lstm_hidden_size + self.prev_action_size
        else:
            fc_input_size = self.lstm_hidden_size

        # TODO: shared fc layers for action and visual vector?
        self.action_fc_layers = nn.ModuleList()
        for i in range(len(action_fc_units)):
            if i == 0:
                in_features = fc_input_size
            else:
                in_features = self.action_fc_units[i-1]
            self.action_fc_layers.append(nn.Sequential(nn.Linear(
                in_features=in_features, out_features=self.action_fc_units[i],
                bias=True), nn.ReLU(), nn.Dropout(self.dropout)))
        self.action_logits = nn.Sequential(nn.Linear(
            in_features=self.action_fc_units[i] if len(self.action_fc_layers) >
            0 else fc_input_size, out_features=self.num_actions, bias=True))

        self.visual_fc_layers = nn.ModuleList()
        for i in range(len(visual_fc_units)):
            if i == 0:
                in_features = fc_input_size
            else:
                in_features = self.visual_fc_units[i-1]
            self.visual_fc_layers.append(nn.Sequential(nn.Linear(
                in_features=in_features, out_features=self.visual_fc_units[i],
                bias=True), nn.ReLU(), nn.Dropout(self.dropout)))

    def forward(self, visual_feature, prev_action_embedding):
        lstm_input = torch.cat([visual_feature, prev_action_embedding], dim=1)

        # TODO: adapt this to multiple batch samples
        # TODO: figure out hidden state and how to track hidden state with
        # multiple threads
        hidden_state, cell_state = self.lstm(lstm_input)

        if self.prev_action_after_lstm:
            fc_input = torch.cat([hidden_state, prev_action_embedding], dim=1)
        else:
            fc_input = hidden_state

        action_fc_output = fc_input
        for action_fc_layer in self.action_fc_layers:
            action_fc_output = action_fc_layer(action_fc_output)
        action_fc_output = self.action_logits(action_fc_output)

        visual_fc_output = fc_input
        for visual_fc_layer in self.visual_fc_layers:
            visual_fc_output = visual_fc_layer(visual_fc_output)

        return action_fc_output, visual_fc_output

class SuperpixelFusion(nn.Module):
    def __init__(self, visual_model=None, superpixel_model=None,
            action_embeddings=None, policy_model=None, slic_kwargs={},
            boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
            black_outer=False):

        """
        neighbor_connectivity of 1 means don't include diagonal adjacency, 2
        means do include diagonal adjacency
        """
        super(SuperpixelFusion, self).__init__()
        self.visual_model = visual_model
        self.superpixel_model = superpixel_model
        self.action_embeddings = action_embeddings
        self.policy_model = policy_model
        self.slic_kwargs = slic_kwargs
        self.boundary_pixels = boundary_pixels
        self.neighbor_depth = neighbor_depth
        self.neighbor_connectivity = neighbor_connectivity
        self.black_outer = black_outer

    def forward(self, frame, last_action):
        """
        Assumes frame is already a torch tensor of floats and moved to GPU.
        """
        visual_features = self.visual_model(frame)
        action, policy_visual_vector = self.policy_model(visual_features,
                self.action_embeddings(last_action))

        superpixel_features = self.get_superpixel_features(frame)

    def get_superpixel_features(self, frame):
        # slic works fine if frame is torch.Tensor
        segments = slic(img_as_float(frame), **self.slic_kwargs)

        superpixel_labels = np.unique(segments)

        superpixel_bounding_boxes = []
        superpixel_masks = []
        if self.neighbor_depth > 0:
            rag = graph.RAG(label_image=segments,
                    connectivity=self.neighbor_connectivity)
            rag_adj = rag.adj # Keep this in case it's an expensive operation
        for label in superpixel_labels:
            labels = [label]
            if self.neighbor_depth > 0:
                labels.extend(list(rag_adj[label].keys()))

            # Get indexes of elements for each label. Row-major order means
            # that rows are ys, columns are xs
            # Have to use np.equal because segments == label returns a single
            # boolean for whatever reason
            mask = reduce(np.logical_or, [segments == label_i for label_i in
                labels])
            ys, xs = np.nonzero(mask)
            max_y = min(frame.shape[0], np.max(ys) + self.boundary_pixels + 1)
            min_y = max(0, np.min(ys) - self.boundary_pixels)
            max_x = min(frame.shape[1], np.max(xs) + self.boundary_pixels + 1)
            min_x = max(0, np.min(xs) - self.boundary_pixels)
            superpixel_bounding_boxes.append((min_y, max_y, min_x, max_x))
            if self.black_outer:
                superpixel_masks.append(mask[min_y:max_y, min_x:max_x])

        frame_crops = [frame[min_y:max_y, min_x:max_x, :] for (min_y, max_y,
            min_x, max_x) in superpixel_bounding_boxes]
        if self.black_outer:
            # Copy frames so the shared frame is not blacked out all over
            copied_frame_crops = []
            for i in range(len(frame_crops)):
                copied_frame_crop = np.copy(frame_crops[i])
                copied_frame_crop[np.logical_not(superpixel_masks[i])] = 0
                copied_frame_crops.append(copied_frame_crop)
            frame_crops = copied_frame_crops

        superpixel_features = self.superpixel_model(frame_crops)

        return superpixel_features


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
    from models.nn.resnet import Resnet
    from skimage import io
    slic_kwargs = {
            'max_iter' : 10,
            'spacing' : None,
            'multichannel' : True,
            'convert2lab' : True,
            'enforce_connectivity' : True,
            'max_size_factor' : 3,
            'n_segments' : 10,
            'compactness' : 10.0,
            'sigma' : 0,
            'min_size_factor' : 0.01
    }
    resnet_args = Namespace(visual_model='resnet', gpu=0)
    resnet = Resnet(resnet_args, use_conv_feat=False)
    sf = SuperpixelFusion(superpixel_model=resnet, slic_kwargs=slic_kwargs,
            neighbor_depth=0, black_outer=True)

    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
    # for alfred's graph in env/tasks.py
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
    import json
    from env.thor_env import ThorEnv
    from env.interaction_exploration import (InteractionExploration,
            InteractionReward)
    env = ThorEnv()
    with open(os.path.join(os.environ['ALFRED_ROOT'], 'models', 'config',
        'rewards.json'), 'r') as jsonfile:
        reward_config = json.load(jsonfile)['InteractionExploration']
    reward = InteractionReward(env, reward_config)
    ie = InteractionExploration(env, reward)
    frame = ie.reset()

    superpixel_features = sf.get_superpixel_features(frame)
    print(superpixel_features.shape)


