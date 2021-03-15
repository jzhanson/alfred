import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json

import gen.constants as constants
from env.interaction_exploration import InteractionExploration

from args import parse_args


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
    from env.thor_env import ThorEnv
    from env.reward import InteractionReward
    from models.nn.resnet import Resnet
    from models.nn.ie import LSTMPolicy, SuperpixelFusion

    args = parse_args()

    thor_env = ThorEnv()

    with open(args.reward_config_path, 'r') as jsonfile:
        reward_config = json.load(jsonfile)['InteractionExploration']


    reward = InteractionReward(thor_env, reward_config,
            reward_rotations=args.reward_rotations,
            reward_look_angle=args.reward_look_angle,
            reward_state_changes=args.reward_state_changes)

    interaction_exploration = InteractionExploration(thor_env, reward,
            single_interact=args.single_interact,
            sample_contextual_action=args.sample_contextual_action,
            use_masks=args.use_masks)

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    slic_kwargs = {
            'max_iter' : args.slic_max_iter,
            'spacing' : None,
            'multichannel' : True,
            'convert2lab' : True,
            'enforce_connectivity' : True,
            'max_size_factor' : args.slic_max_size_factor,
            'n_segments' : args.slic_n_segments,
            'compactness' : args.slic_compactness,
            'sigma' : args.slic_sigma,
            'min_size_factor' : args.slic_min_size_factor
    }

    if args.single_interact:
        num_actions = len(constants.SIMPLE_ACTIONS)
    else:
        num_actions = len(constants.COMPLEX_ACTIONS)

    action_embeddings = nn.Embedding(num_embeddings=num_actions,
            embedding_dim=args.action_embedding_dim)

    if 'resnet' in args.visual_model:
        args.visual_feature_size = 512
        resnet_args = Namespace(visual_model='resnet', gpu=args.gpu)
        visual_model = Resnet(resnet_args, use_conv_feat=False)
    else:
        print("visual model '" + args.visual_model + "' not supported")

    if 'resnet' in args.superpixel_model:
        args.superpixel_feature_size = 512
        resnet_args = Namespace(visual_model='resnet', gpu=args.gpu)
        superpixel_model = Resnet(resnet_args, use_conv_feat=False)
    else:
        print("superpixel model '" + args.superpixel_model + "' not supported")

    if type(args.action_fc_units) is int:
        args.action_fc_units = [args.action_fc_units]
    if type(args.visual_fc_units) is int:
        args.visual_fc_units = [args.visual_fc_units]

    policy_model = LSTMPolicy(num_actions=num_actions,
            visual_feature_size=args.visual_feature_size,
            superpixel_feature_size=args.superpixel_feature_size,
            prev_action_size=args.action_embedding_dim,
            lstm_hidden_size=args.lstm_hidden_dim, dropout=args.dropout,
            action_fc_units=args.action_fc_units,
            visual_fc_units=args.visual_fc_units,
            prev_action_after_lstm=args.prev_action_after_lstm)

    sf = SuperpixelFusion(action_embeddings=action_embeddings,
          visual_model=visual_model, superpixel_model=superpixel_model,
          policy_model=policy_model, slic_kwargs=slic_kwargs,
          boundary_pixels=args.boundary_pixels,
          neighbor_depth=args.neighbor_depth,
          neighbor_connectivity=args.neighbor_connectivity, black_outer=False)

