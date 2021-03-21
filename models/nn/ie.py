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

import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
from models.nn.resnet import Resnet

class SingleLayerCNN(nn.Module):
    def __init__(self, input_width=300, input_height=300, output_size=64):
        super(SingleLayerCNN, self).__init__()

    def forward(self, frame):
        pass


class LSTMPolicy(nn.Module):
    def __init__(self, num_actions=12, visual_feature_size=512,
            superpixel_feature_size=512, prev_action_size=16,
            lstm_hidden_size=512, dropout=0, action_fc_units=[],
            value_fc_units=[], visual_fc_units=[],
            prev_action_after_lstm=False):

        super(LSTMPolicy, self).__init__()

        self.num_actions = num_actions
        self.visual_feature_size = visual_feature_size
        self.superpixel_feature_size = superpixel_feature_size
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

        self.value_fc_layers = nn.ModuleList()
        for i in range(len(value_fc_units)):
            if i == 0:
                in_features = fc_input_size
            else:
                in_features = self.value_fc_units[i-1]
            self.value_fc_layers.append(nn.Sequential(nn.Linear(
                in_features=in_features, out_features=self.value_fc_units[i],
                bias=True), nn.ReLU(), nn.Dropout(self.dropout)))
        self.value = nn.Sequential(nn.Linear(
            in_features=self.value_fc_units[i] if len(self.value_fc_layers) >
            0 else fc_input_size, out_features=1, bias=True))

        self.visual_fc_layers = nn.ModuleList()
        for i in range(len(visual_fc_units)):
            if i == 0:
                in_features = fc_input_size
            else:
                in_features = self.visual_fc_units[i-1]
            self.visual_fc_layers.append(nn.Sequential(nn.Linear(
                in_features=in_features, out_features=self.visual_fc_units[i],
                bias=True), nn.ReLU(), nn.Dropout(self.dropout)))

    def forward(self, visual_feature, prev_action_embedding, policy_hidden):
        lstm_input = torch.cat([visual_feature, prev_action_embedding], dim=1)

        # TODO: adapt this to multiple batch samples
        # TODO: figure out how to track hidden state with multiple threads
        hidden_state, cell_state = self.lstm(lstm_input, policy_hidden)

        if self.prev_action_after_lstm:
            fc_input = torch.cat([hidden_state, prev_action_embedding], dim=1)
        else:
            fc_input = hidden_state

        action_fc_output = fc_input
        for action_fc_layer in self.action_fc_layers:
            action_fc_output = action_fc_layer(action_fc_output)
        action_fc_output = self.action_logits(action_fc_output)

        value_fc_output = fc_input
        for value_fc_layer in self.value_fc_layers:
            value_fc_output = value_fc_layer(value_fc_output)
        value_fc_output = self.value(value_fc_output)

        visual_fc_output = fc_input
        for visual_fc_layer in self.visual_fc_layers:
            visual_fc_output = visual_fc_layer(visual_fc_output)

        return (action_fc_output, value_fc_output, visual_fc_output,
                (hidden_state, cell_state))

    def init_hidden(self, batch_size=1, device=torch.device('cpu')):
        return (torch.zeros(batch_size, self.lstm_hidden_size, device=device),
                torch.zeros(batch_size, self.lstm_hidden_size, device=device))

# TODO: maybe this class should be in a different place since it deals
# with choosing a superpixel more than any neural network stuff
class SuperpixelFusion(nn.Module):
    # TODO: add a better initialization for fc layers and LSTMPolicy
    def __init__(self, visual_model=None, superpixel_model=None,
            action_embeddings=None, policy_model=None, slic_kwargs={},
            boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
            black_outer=False, sample_action=True):

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
        self.sample_action = sample_action

    def forward(self, frame, last_action_index, policy_hidden=None,
            device=torch.device('cpu')):
        """
        Assumes frame is already a torch tensor of floats and moved to GPU.
        """
        # TODO: make this class okay with batched frame and action if necessary
        # for threading
        visual_features = self.featurize(frame)
        #print('visual features', len(visual_features), visual_features[0].shape)

        # TODO: update this so it will work if we're batching multiple
        # last_action_index, consider making last_action_index [batch_size, 1]
        # instead of [batch_size]
        if last_action_index[0] is not None:
            last_action_embedding = self.action_embeddings(last_action_index)
        else:
            # TODO: A bit of a hack to get zeros like an embedding
            last_action_embedding = torch.zeros_like(self.action_embeddings(
                torch.LongTensor([0]).to(device)))

        action_scores, value, visual_output, hidden_state = self.policy_model(
                visual_features, last_action_embedding, policy_hidden)

        #print('action scores', action_scores.shape)

        # TODO: make frame_crops work if frames are stacked
        batch_superpixel_masks = []
        batch_frame_crops = []
        batch_superpixel_features = []
        for i in range(frame.shape[0]):
            superpixel_masks, frame_crops = self.get_superpixel_masks_frame_crops(
                    frame[i])
            batch_superpixel_masks.append(superpixel_masks)
            batch_frame_crops.append(frame_crops)

            # TODO: want to stack frames for superpixels?
            superpixel_features = self.featurize(frame_crops,
                    superpixel_model=True)
            batch_superpixel_features.append(superpixel_features)

        batch_superpixel_features = torch.stack(batch_superpixel_features)
        #print('batch superpixel features', batch_superpixel_features.shape)
        #print('visual output', visual_output.shape)

        # Get rid of last two dimensions since Resnet features are (512, 1, 1)
        if isinstance(self.visual_model, Resnet):
            batch_superpixel_features = torch.squeeze(batch_superpixel_features, -1)
            batch_superpixel_features = torch.squeeze(batch_superpixel_features, -1)

        similarity_scores = torch.sum(visual_output * batch_superpixel_features,
                dim=-1)

        #print('similarity scores', similarity_scores.shape)

        return (action_scores, value, similarity_scores, superpixel_masks,
                hidden_state)

    def featurize(self, stacked_frames, superpixel_model=False):
        chosen_model = (self.superpixel_model if superpixel_model else
                self.visual_model)
        if isinstance(chosen_model, Resnet) and isinstance(stacked_frames,
                torch.Tensor):
            # Unstack frames, featurize, then restack frames if using Resnet
            unstacked_visual_outputs = []
            for unstacked_frames in torch.split(stacked_frames,
                    split_size_or_sections=3, dim=1):
                # TODO: is there a better way to avoid copying 2-3 times?
                # Cast to uint8 first to reduce amount of memory copied. Resnet
                # wants CPU tensors to convert to PIL Image, but frame is a
                # CUDA tensor. We could change stack_frames in
                # models/model/supervised_find.py to not convert frames to
                # CUDA, but we still copy for now to keep the interface that
                # visual_model expects CUDA tensors for everything and also so
                # these models require as little knowledge about device as
                # possible.
                unstacked_frames = [frame.to(dtype=torch.uint8).cpu() for
                        frame in unstacked_frames]
                unstacked_visual_outputs.append(chosen_model(unstacked_frames))
            # unstacked_visual_outputs is now length frame_stack, each element
            # is shape (batch_size, 3, 300, 300)
            output = torch.cat(unstacked_visual_outputs, dim=1)
        else:
            #print('stacked frames', len(stacked_frames),
            #        stacked_frames[0].shape, stacked_frames[0])
            for i in range(len(stacked_frames)):
                stacked_frames[i] = np.ascontiguousarray(stacked_frames[i]).astype('uint8')
            output = chosen_model(stacked_frames)

        # Flatten latter dimensions of visual output in case it's made up of
        # conv features
        output = torch.flatten(output, start_dim=1)
        return output

    def get_superpixel_masks_frame_crops(self, frame):
        """
        Returns superpixel masks for each superpixel over the whole image and
        frame crops of the original image cropped to the bounding box of the
        superpixel.
        """
        # slic works fine if frame is torch.Tensor
        # Need to reshape frame from [3, 300, 300] to [300, 300, 3]
        frame = frame.numpy().transpose(1, 2, 0)
        segments = slic(img_as_float(frame), **self.slic_kwargs)

        superpixel_labels = np.unique(segments)

        superpixel_bounding_boxes = []
        superpixel_masks = []
        if self.neighbor_depth > 0:
            rag = graph.RAG(label_image=segments,
                    connectivity=self.neighbor_connectivity)
            rag_adj = rag.adj # Keep this in case it's an expensive operation
        # The original segment labels don't matter since we only use
        # superpixel_bounding_boxes and superpixel_masks, which match up
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
            superpixel_masks.append(mask)

        frame_crops = [frame[min_y:max_y, min_x:max_x, :] for (min_y, max_y,
            min_x, max_x) in superpixel_bounding_boxes]
        if self.black_outer:
            # Copy frames so the shared frame is not blacked out all over
            copied_frame_crops = []
            for i in range(len(frame_crops)):
                copied_frame_crop = np.copy(frame_crops[i])
                min_y, max_y, min_x, max_x = superpixel_bounding_boxes[i]
                cropped_mask = superpixel_masks[i][min_y:max_y, min_x:max_x]
                copied_frame_crop[np.logical_not(cropped_mask)] = 0
                copied_frame_crops.append(copied_frame_crop)
            frame_crops = copied_frame_crops

        return superpixel_masks, frame_crops

    def init_policy_hidden(self, batch_size=1, device=torch.device('cpu')):
        if isinstance(self.policy_model, LSTMPolicy):
            return self.policy_model.init_hidden(batch_size=batch_size,
                    device=device)
        return None

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == '__main__':
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
    action_embeddings = nn.Embedding(num_embeddings=12, embedding_dim=16)
    policy_model = LSTMPolicy()
    resnet_args = Namespace(visual_model='resnet', gpu=0)
    visual_model = Resnet(resnet_args, use_conv_feat=False)
    superpixel_model = Resnet(resnet_args, use_conv_feat=False)
    sf = SuperpixelFusion(action_embeddings=action_embeddings,
            visual_model=visual_model, superpixel_model=superpixel_model,
            policy_model=policy_model, slic_kwargs=slic_kwargs,
            neighbor_depth=0, black_outer=True)

    '''
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
    '''

    frame = io.imread('/home/jzhanson/alfred/saved/test_frame.png')
    masks, frame_crops = sf.forward(frame, 0)
    print(masks)
    print(frame_crops)


