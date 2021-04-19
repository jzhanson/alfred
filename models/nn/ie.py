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
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
from models.nn.resnet import Resnet
import gen.constants as constants

def init_fc_layers(fc_units, input_size, use_tanh=True, dropout=0.0,
        last_activation=False):
    fc_layers = nn.ModuleList()
    for i in range(len(fc_units)):
        if i == 0:
            in_features = input_size
        else:
            in_features = fc_units[i-1]
        # No activation or dropout on last layer
        if i == len(fc_units) - 1 and not last_activation:
            fc_layers.append(nn.Sequential(nn.Linear(in_features=in_features,
                out_features=fc_units[i], bias=True)))
        else:
            fc_layers.append(nn.Sequential(nn.Linear(in_features=in_features,
                out_features=fc_units[i], bias=True), nn.Tanh() if use_tanh
                else nn.ReLU(), nn.Dropout(dropout)))
    return fc_layers

class SingleLayerCNN(nn.Module):
    def __init__(self, input_width=300, input_height=300, output_size=64):
        super(SingleLayerCNN, self).__init__()

    def forward(self, frame):
        pass


class LSTMPolicy(nn.Module):
    def __init__(self, visual_feature_size=512,
            prev_action_size=16, lstm_hidden_size=512, dropout=0,
            action_fc_units=[], value_fc_units=[], visual_fc_units=[],
            prev_action_after_lstm=False, use_tanh=True):
        """
        If unified action space (e.g. similarity vector to choose concatenated
        superpixel + action embedding), can leave action_fc_units or
        visual_fc_units as [] and only use one or the other.
        """

        super(LSTMPolicy, self).__init__()

        self.visual_feature_size = visual_feature_size
        self.prev_action_size = prev_action_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = dropout
        self.action_fc_units = action_fc_units
        self.value_fc_units = value_fc_units
        self.visual_fc_units = visual_fc_units
        self.prev_action_after_lstm = prev_action_after_lstm
        self.use_tanh = use_tanh

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
        self.action_fc_layers = init_fc_layers(self.action_fc_units,
                fc_input_size, use_tanh=self.use_tanh, dropout=self.dropout,
                last_activation=False)

        self.value_fc_layers = init_fc_layers(self.value_fc_units,
                fc_input_size, use_tanh=self.use_tanh, dropout=self.dropout,
                last_activation=True)
        self.value = nn.Sequential(nn.Linear(
            in_features=self.value_fc_units[-1] if len(self.value_fc_layers) >
            0 else fc_input_size, out_features=1, bias=True))

        self.visual_fc_layers = init_fc_layers(self.visual_fc_units,
                fc_input_size, use_tanh=self.use_tanh, dropout=self.dropout,
                last_activation=False)

    def forward(self, visual_feature, prev_action_feature, policy_hidden):
        #print('visual feature', visual_feature.shape)
        #print('prev action feature', prev_action_feature.shape)
        lstm_input = torch.cat([visual_feature, prev_action_feature], dim=1)

        hidden_state, cell_state = self.lstm(lstm_input, policy_hidden)

        if self.prev_action_after_lstm:
            fc_input = torch.cat([hidden_state, prev_action_feature], dim=1)
        else:
            fc_input = hidden_state

        action_fc_output = fc_input
        for action_fc_layer in self.action_fc_layers:
            action_fc_output = action_fc_layer(action_fc_output)

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

class ResnetSuperpixelWrapper(nn.Module):
    """
    Class to wrap Resnet classes/models to put an extra fc layer + tanh
    activation to make the features output for superpixels the same magnitude
    as the visual vector output from the model.

    Requires superpixel_model to have superpixel_model.output_size.
    """
    def __init__(self, superpixel_model=None, superpixel_fc_units=[],
            dropout=0.0, use_tanh=True):
        super(ResnetSuperpixelWrapper, self).__init__()
        self.superpixel_model = superpixel_model
        self.superpixel_fc_units = superpixel_fc_units
        self.dropout = dropout
        self.use_tanh = use_tanh

        self.superpixel_fc_layers = init_fc_layers(self.superpixel_fc_units,
                self.superpixel_model.output_size, use_tanh=self.use_tanh,
                dropout=self.dropout, last_activation=True)
        self.output_size = self.superpixel_fc_units[-1]

    def forward(self, superpixel_crop):
        superpixel_fc_output = self.superpixel_model(superpixel_crop)
        # We need to squeeze out the last two dimensions of the Resnet features
        # (N, 512, 1, 1) -> (N, 512)
        superpixel_fc_output = superpixel_fc_output.squeeze(-1).squeeze(-1)
        for superpixel_fc_layer in self.superpixel_fc_layers:
            superpixel_fc_output = superpixel_fc_layer(superpixel_fc_output)
        return superpixel_fc_output

class SuperpixelFusion(nn.Module):
    # TODO: add a better initialization for fc layers and LSTMPolicy
    def __init__(self, visual_model=None, superpixel_model=None,
            action_embeddings=None, policy_model=None, slic_kwargs={},
            boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
            black_outer=False, device=torch.device('cpu')):
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

    def forward(self, frame, last_action_features, policy_hidden=None,
            gt_segmentation=None, device=torch.device('cpu')):
        """
        Assumes frame is already a torch tensor of floats and moved to GPU.
        last_action_index is a list of length batch_size of tensors or Nones
        (to signal there being no previous action). We don't make it a list of
        length batch_size and wrap the inner indexes with an extra dimension
        (in essence [batch_size, 1] instead of [batch_size]) because of the
        Nones - it would require extra processing "for no reason"
        """
        visual_features = self.featurize(frame)

        action_scores, value, visual_output, hidden_state = self.policy_model(
                visual_features, last_action_features, policy_hidden)

        #print('action scores', action_scores.shape)
        batch_size = frame.shape[0]
        batch_action_features = []
        batch_superpixel_masks = []
        batch_frame_crops = []
        batch_superpixel_features = []
        batch_similarity_scores = []
        for i in range(batch_size):
            batch_action_features.append(self.action_embeddings(
                torch.LongTensor([i for i in range(action_scores.shape[1])])
                .to(device)))
            # Take last three channels (RGB of last stacked frame)
            if gt_segmentation is not None:
                superpixel_masks, frame_crops = (
                        self.get_gt_segmentation_masks_frame_crops(
                            frame[i][-3:], gt_segmentation))
            else:
                superpixel_masks, frame_crops = (
                        self.get_superpixel_masks_frame_crops(frame[i][-3:]))
            batch_superpixel_masks.append(superpixel_masks)
            batch_frame_crops.append(frame_crops)

            # Stacking frames for superpixels doesn't seem like a good idea,
            # since moving and looking around changes the view so drastically
            superpixel_features = self.featurize(frame_crops,
                    superpixel_model=True)
            # Get rid of last two dimensions since Resnet features are (512, 1, 1)
            # We only need to do this if it's a raw Resnet and not a
            # ResnetSuperpixelWrapper
            if isinstance(self.visual_model, Resnet):
                superpixel_features = torch.squeeze(superpixel_features, -1)
                superpixel_features = torch.squeeze(superpixel_features, -1)
            batch_superpixel_features.append(superpixel_features)
            batch_similarity_scores.append(torch.sum(visual_output *
                superpixel_features, dim=-1))

        # Because each batch can have different numbers of superpixels,
        # batch_similarity_scores and batch_superpixel_masks have to be lists
        # of tensors instead of tensors
        return (action_scores, value, batch_similarity_scores,
                batch_superpixel_masks, (batch_action_features,
                    batch_superpixel_features), hidden_state)

    def featurize(self, stacked_frames, superpixel_model=False):
        chosen_model = (self.superpixel_model if superpixel_model else
                self.visual_model)
        if (isinstance(chosen_model, Resnet) or isinstance(chosen_model,
            ResnetSuperpixelWrapper)) and isinstance(stacked_frames,
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
        # Cast to uint8 since the slic code behaves differently for float32
        frame = frame.numpy().transpose(1, 2, 0).astype('uint8')
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
            max_y, min_y, max_x, min_x = (SuperpixelFusion
                    .get_max_min_y_x_with_boundary(frame, ys, xs,
                        self.boundary_pixels))
            superpixel_bounding_boxes.append((min_y, max_y, min_x, max_x))
            superpixel_masks.append(mask)

        frame_crops = [frame[min_y:max_y, min_x:max_x, :] for (min_y, max_y,
            min_x, max_x) in superpixel_bounding_boxes]
        if self.black_outer:
            frame_crops = SuperpixelFusion.get_black_outer_frame_crops(
                    frame_crops, superpixel_bounding_boxes, superpixel_masks)

        return superpixel_masks, frame_crops

    def get_gt_segmentation_masks_frame_crops(self, frame, segmentation):
        """
        Applies self.boundary_pixels and self.black_outer but not
        self.neighbor_connectivity.
        """
        # Still reshape frame from [3, 300, 300] to [300, 300, 3], for
        # consistency with get_superpixel_masks_frame_crops
        frame = frame.numpy().transpose(1, 2, 0).astype('uint8')

        color_to_pixel_yxs = {}
        for y in range(segmentation.shape[0]):
            for x in range(segmentation.shape[1]):
                color = tuple(segmentation[y, x])
                if color not in color_to_pixel_yxs:
                    color_to_pixel_yxs[color] = []
                color_to_pixel_yxs[color].append((y, x))
        bounding_boxes = []
        masks = []

        for _, pixel_yxs in color_to_pixel_yxs.items():
            ys = [y for (y, x) in pixel_yxs]
            xs = [x for (y, x) in pixel_yxs]
            max_y, min_y, max_x, min_x = (SuperpixelFusion
                    .get_max_min_y_x_with_boundary(frame, ys, xs,
                        self.boundary_pixels))
            bounding_boxes.append((min_y, max_y, min_x, max_x))
            mask = np.zeros((frame.shape[0], frame.shape[1]))
            # Multidimensional/tuple indexing requires a list of first
            # dimension coordinates, then a list of second dimension
            # coordinates
            # https://stackoverflow.com/questions/28491230/indexing-a-numpy-array-with-a-list-of-tuples
            mask[tuple(zip(*pixel_yxs))] = 1
            masks.append(mask)

        frame_crops = [frame[min_y:max_y, min_x:max_x, :] for (min_y, max_y,
            min_x, max_x) in bounding_boxes]

        if self.black_outer:
            frame_crops = SuperpixelFusion.get_black_outer_frame_crops(
                    frame_crops, bounding_boxes, masks)

        return masks, frame_crops

    def init_policy_hidden(self, batch_size=1, device=torch.device('cpu')):
        if isinstance(self.policy_model, LSTMPolicy):
            return self.policy_model.init_hidden(batch_size=batch_size,
                    device=device)
        return None

    @classmethod
    def get_max_min_y_x_with_boundary(cls, frame, ys, xs, boundary_pixels):
        max_y = min(frame.shape[0], np.max(ys) + boundary_pixels + 1)
        min_y = max(0, np.min(ys) - boundary_pixels)
        max_x = min(frame.shape[1], np.max(xs) + boundary_pixels + 1)
        min_x = max(0, np.min(xs) - boundary_pixels)
        return max_y, min_y, max_x, min_x

    @classmethod
    def get_black_outer_frame_crops(cls, frame_crops, bounding_boxes, masks):
        # Copy frames so the shared frame is not blacked out all over
        copied_frame_crops = []
        for i in range(len(frame_crops)):
            copied_frame_crop = np.copy(frame_crops[i])
            min_y, max_y, min_x, max_x = bounding_boxes[i]
            cropped_mask = masks[i][min_y:max_y, min_x:max_x]
            copied_frame_crop[np.logical_not(cropped_mask)] = 0
            copied_frame_crops.append(copied_frame_crop)
        return copied_frame_crops

class SuperpixelActionConcat(SuperpixelFusion):
    def __init__(self, visual_model=None, superpixel_model=None,
            action_embeddings=None, policy_model=None, slic_kwargs={},
            boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
            black_outer=False, superpixel_feature_size=512,
            single_interact=False, zero_null_superpixel_features=False,
            device=torch.device('cpu')):
        super(SuperpixelActionConcat, self).__init__(visual_model=visual_model,
                superpixel_model=superpixel_model,
                action_embeddings=action_embeddings, policy_model=policy_model,
                slic_kwargs=slic_kwargs, boundary_pixels=boundary_pixels,
                neighbor_depth=neighbor_depth,
                neighbor_connectivity=neighbor_connectivity,
                black_outer=black_outer, device=device)
        self.superpixel_feature_size = superpixel_feature_size
        self.single_interact = single_interact
        self.zero_null_superpixel_features = zero_null_superpixel_features

    def forward(self, frame, last_action_features, policy_hidden=None,
            gt_segmentation=None, device=torch.device('cpu')):
        """
        Nav actions always come at the beginning of the returned scores,
        followed by interact actions.
        """
        visual_features = self.featurize(frame)

        action_output, value, _, hidden_state = self.policy_model(
                visual_features, last_action_features, policy_hidden)

        batch_size = frame.shape[0]
        batch_superpixel_masks = []
        batch_frame_crops = []
        batch_superpixel_features = [] # Don't really need this
        # Passing the mask should be by reference, so masks aren't being copied
        # all the time
        batch_concatenated_features = [] # Don't really need this either
        batch_actions_masks_features = []
        batch_similarity_scores = []
        if self.single_interact:
            interact_actions = [constants.ACTIONS_INTERACT]
        else:
            interact_actions = constants.INT_ACTIONS
        for batch_i in range(batch_size):
            # Take last three channels (RGB of last stacked frame)
            if gt_segmentation is not None:
                superpixel_masks, frame_crops = (
                        self.get_gt_segmentation_masks_frame_crops(
                            frame[batch_i][-3:], gt_segmentation))
            else:
                superpixel_masks, frame_crops = (
                        self.get_superpixel_masks_frame_crops(
                            frame[batch_i][-3:]))
            batch_superpixel_masks.append(superpixel_masks)
            batch_frame_crops.append(frame_crops)

            # Stacking frames for superpixels doesn't seem like a good idea,
            # since moving and looking around changes the view so drastically
            superpixel_features = self.featurize(frame_crops,
                    superpixel_model=True)

            # Get rid of last two dimensions since Resnet features are (512, 1,
            # 1)
            # Again, we only need to do this if it's a raw Resnet and not a
            # ResnetSuperpixelWrapper
            if isinstance(self.visual_model, Resnet):
                superpixel_features = torch.squeeze(superpixel_features, -1)
                superpixel_features = torch.squeeze(superpixel_features, -1)
            batch_superpixel_features.append(superpixel_features)

            concatenated_features = []
            actions_masks_features = []
            if self.zero_null_superpixel_features:
                null_superpixel_features = torch.zeros_like(
                        batch_superpixel_features[0][0])
            else:
                null_superpixel_features = torch.mean(superpixel_features,
                        dim=0)

            for action_i in range(len(constants.NAV_ACTIONS)):
                # action_embeddings takes in (batch_size) and returns
                # (batch_size, embedding_dim)
                concatenated_feature = torch.cat([
                    self.action_embeddings(torch.LongTensor([action_i])
                        .to(device)).squeeze(0),
                    null_superpixel_features])
                concatenated_features.append(concatenated_feature)
                actions_masks_features.append((constants.NAV_ACTIONS[action_i],
                    None, concatenated_feature))
            for action_i in range(len(interact_actions)):
                for superpixel_i in range(superpixel_features.shape[0]):
                    concatenated_feature = torch.cat([
                        self.action_embeddings(torch.LongTensor([action_i])
                            .to(device)).squeeze(0),
                        superpixel_features[superpixel_i]])
                    concatenated_features.append(concatenated_feature)
                    actions_masks_features.append((interact_actions[action_i],
                        superpixel_masks[superpixel_i], concatenated_feature))
            concatenated_features = torch.stack(concatenated_features)
            batch_concatenated_features.append(concatenated_features)
            batch_actions_masks_features.append(actions_masks_features)
            batch_similarity_scores.append(torch.sum(action_output *
                concatenated_features, dim=-1))

        # Because each batch can have different numbers of superpixels,
        # batch_similarity_scores and batch_action_mask_index_pairs have to be
        # lists of tensors instead of tensors
        return (action_output, value, batch_similarity_scores,
                batch_actions_masks_features, hidden_state)

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
    policy_model = LSTMPolicy(prev_action_size=16, action_fc_units=[12])
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
    print(frame.shape)

    print('SuperpixelFusion')
    (action_scores, value, batch_similarity_scores, batch_superpixel_masks,
            (batch_action_features, batch_mask_features),
            hidden_state) = sf.forward(torch.Tensor([frame.transpose(2, 0,
                1)]), torch.zeros(1, 16))
    print('action_scores', action_scores.shape)
    print('value', value.shape)
    print('batch_similarity_scores', len(batch_similarity_scores),
            batch_similarity_scores[0].shape)
    print('batch_superpixel_masks', len(batch_superpixel_masks))
    print('batch_action_features', len(batch_action_features),
            batch_action_features[0].shape)
    print('batch_mask_features', len(batch_mask_features))

    print('SuperpixelActionConcat')
    policy_model = LSTMPolicy(prev_action_size=512+16,
            action_fc_units=[512+16])
    model = SuperpixelActionConcat(action_embeddings=action_embeddings,
            visual_model=visual_model, superpixel_model=superpixel_model,
            policy_model=policy_model, slic_kwargs=slic_kwargs,
            neighbor_depth=0, black_outer=True, superpixel_feature_size=512,
            single_interact=False, zero_null_superpixel_features=False)

    (action_output, value, batch_similarity_scores,
                batch_actions_masks_features, hidden_state) = model.forward(
                        torch.Tensor([frame.transpose(2, 0, 1)]),
                        torch.cat([action_embeddings(torch.LongTensor([0])),
                            torch.zeros((1, 512))], dim=1))
    print('action_output', action_output.shape)
    print('value', value.shape)
    print('batch similarity scores', len(batch_similarity_scores),
            batch_similarity_scores[0].shape)
    print('batch_actions_masks_features', len(batch_actions_masks_features),
            len(batch_actions_masks_features[0]),
            batch_actions_masks_features[0][0])

