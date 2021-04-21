import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import torch
import torch.nn as nn
import torch.optim as optim

import gen.constants as constants
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
from env.thor_env import ThorEnv
from env.interaction_exploration import InteractionExploration
from env.reward import InteractionReward
from models.model.args import parse_args
from models.nn.resnet import Resnet
from models.nn.ie import (CuriosityIntrinsicReward, LSTMPolicy,
        ResnetSuperpixelWrapper, SuperpixelFusion, SuperpixelActionConcat)
from models.model.rl_interaction import train

if __name__ == '__main__':
    args = parse_args()

    thor_env = ThorEnv()

    with open(os.path.join(os.environ['ALFRED_ROOT'], 'models/config/rewards.json'),
            'r') as jsonfile:
        reward_config = json.load(jsonfile)[args.reward_config_name]

    # TODO: if adding threads, add support for running on multiple gpus, e.g.
    # gpu_ids like
    # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/train.py
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    reward = InteractionReward(thor_env, reward_config,
            reward_rotations_look_angles=args.reward_rotations_look_angles,
            reward_state_changes=args.reward_state_changes,
            persist_state=args.reward_persist_state,
            repeat_discount=args.reward_repeat_discount,
            use_novelty=args.reward_use_novelty)

    ie = InteractionExploration(thor_env, reward,
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
            embedding_dim=args.action_embedding_dim).to(device)

    resnet_args = Namespace(visual_model='resnet', gpu=args.gpu >= 0,
            gpu_index=args.gpu)
    if 'resnet' in args.visual_model:
        args.visual_feature_size = 512
        visual_model = Resnet(resnet_args, use_conv_feat=False,
                pretrained=args.pretrained_visual_model)
    else:
        print("visual model '" + args.visual_model + "' not supported")

    if 'resnet' in args.superpixel_model:
        args.superpixel_feature_size = 512
        superpixel_model = Resnet(resnet_args, use_conv_feat=False,
                pretrained=args.pretrained_visual_model)
    else:
        print("superpixel model '" + args.superpixel_model + "' not supported")

    if args.superpixel_fc_units is None:
        args.superpixel_fc_units = []
    elif type(args.superpixel_fc_units) is int:
        args.superpixel_fc_units = [args.superpixel_fc_units]
    superpixel_model = ResnetSuperpixelWrapper(
            superpixel_model=superpixel_model,
            superpixel_fc_units=args.superpixel_fc_units, dropout=args.dropout,
            use_tanh=args.use_tanh)

    if args.action_fc_units is None:
        args.action_fc_units = []
    elif type(args.action_fc_units) is int:
        args.action_fc_units = [args.action_fc_units]
    if args.value_fc_units is None:
        args.value_fc_units = []
    elif type(args.value_fc_units) is int:
        args.value_fc_units = [args.value_fc_units]
    if args.visual_fc_units is None:
        args.visual_fc_units = []
    elif type(args.visual_fc_units) is int:
        args.visual_fc_units = [args.visual_fc_units]

    if args.fusion_model == 'SuperpixelFusion':
        policy_model = LSTMPolicy(
                visual_feature_size=args.visual_feature_size,
                prev_action_size=args.action_embedding_dim +
                    args.superpixel_feature_size,
                lstm_hidden_size=args.lstm_hidden_dim, dropout=args.dropout,
                action_fc_units=args.action_fc_units,
                value_fc_units=args.value_fc_units,
                visual_fc_units=args.visual_fc_units,
                prev_action_after_lstm=args.prev_action_after_lstm,
                use_tanh=args.use_tanh).to(device)

        model = SuperpixelFusion(action_embeddings=action_embeddings,
              visual_model=visual_model, superpixel_model=superpixel_model,
              policy_model=policy_model, slic_kwargs=slic_kwargs,
              boundary_pixels=args.boundary_pixels,
              neighbor_depth=args.neighbor_depth,
              neighbor_connectivity=args.neighbor_connectivity,
              black_outer=args.black_outer, device=device)
    elif args.fusion_model == 'SuperpixelActionConcat':
        vector_size = args.superpixel_feature_size + args.action_embedding_dim
        policy_model = LSTMPolicy(
                visual_feature_size=args.visual_feature_size,
                prev_action_size=vector_size,
                lstm_hidden_size=args.lstm_hidden_dim, dropout=args.dropout,
                action_fc_units=args.action_fc_units,
                value_fc_units=args.value_fc_units,
                visual_fc_units=args.visual_fc_units,
                prev_action_after_lstm=args.prev_action_after_lstm).to(device)

        model = SuperpixelActionConcat(action_embeddings=action_embeddings,
              visual_model=visual_model, superpixel_model=superpixel_model,
              policy_model=policy_model, slic_kwargs=slic_kwargs,
              boundary_pixels=args.boundary_pixels,
              neighbor_depth=args.neighbor_depth,
              neighbor_connectivity=args.neighbor_connectivity,
              black_outer=args.black_outer,
              single_interact=args.single_interact,
              zero_null_superpixel_features=args.zero_null_superpixel_features,
              device=device)

    try:
        model = model.to(device)
    except:
        model = model.to(device)

    if args.use_curiosity:
        if 'resnet' in args.curiosity_visual_encoder:
            curiosity_visual_encoder = Resnet(resnet_args, use_conv_feat=False,
                    pretrained=args.pretrained_visual_model)
        else:
            print("curiosity visual encoder '" + args.curiosity_visual_encoder
                    + "' not supported")

        if args.curiosity_forward_fc_units is None:
            args.curiosity_forward_fc_units = []
        elif type(args.curiosity_forward_fc_units) is int:
            args.curiosity_forward_fc_units = [args.curiosity_forward_fc_units]
        if args.curiosity_inverse_fc_units is None:
            args.curiosity_inverse_fc_units = []
        elif type(args.curiosity_inverse_fc_units) is int:
            args.curiosity_inverse_fc_units = [args.curiosity_inverse_fc_units]

        curiosity_model = CuriosityIntrinsicReward(
                visual_encoder=curiosity_visual_encoder,
                action_embedding_dim=args.action_embedding_dim +
                    args.superpixel_feature_size,
                forward_fc_units=args.curiosity_forward_fc_units,
                inverse_fc_units=args.curiosity_inverse_fc_units,
                eta=args.curiosity_eta, beta=args.curiosity_beta,
                use_tanh=args.use_tanh, dropout=args.dropout)
        try:
            curiosity_model = curiosity_model.to(device)
        except:
            curiosity_model = curiosity_model.to(device)
    else:
        curiosity_model = None

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    if 'adam' in args.optimizer:
        amsgrad = 'amsgrad' in args.optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=amsgrad)

    print('model parameters: ' + str(sum(p.numel() for p in model.parameters()
        if p.requires_grad)))

    scene_numbers = []
    if args.scene_numbers is not None:
        if type(args.scene_numbers) is int:
            scene_numbers = [args.scene_numbers]
        else:
            scene_numbers = args.scene_numbers
    else:
        # "'str' in" pattern will work for both list and single string
        if 'kitchen' in args.scene_types:
            scene_numbers.extend(constants.KITCHEN_TRAIN_SCENE_NUMBERS)
        elif 'living_room' in args.scene_types:
            scene_numbers.extend(constants.LIVING_ROOM_TRAIN_SCENE_NUMBERS)
        elif 'bedroom' in args.scene_types:
            scene_numbers.extend(constants.BEDROOM_TRAIN_SCENE_NUMBERS)
        elif 'bathroom' in args.scene_types:
            scene_numbers.extend(constants.BATHROOM_TRAIN_SCENE_NUMBERS)

    train(model, ie, optimizer, gamma=args.gamma, tau=args.tau,
            value_loss_coefficient=args.value_loss_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            max_grad_norm=args.max_grad_norm,
            single_interact=args.single_interact, use_masks=args.use_masks,
            use_gt_segmentation=args.use_gt_segmentation,
            fusion_model=args.fusion_model,
            outer_product_sampling=args.outer_product_sampling,
            inverse_score=args.inverse_score,
            zero_null_superpixel_features=args.zero_null_superpixel_features,
            curiosity_model=curiosity_model,
            curiosity_lambda=args.curiosity_lambda,
            scene_numbers=scene_numbers,
            max_trajectory_length=args.max_trajectory_length,
            frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            teacher_force=args.teacher_force, sample_action=args.sample_action,
            sample_mask=args.sample_mask, train_episodes=args.train_episodes,
            valid_seen_episodes=args.valid_seen_episodes,
            valid_unseen_episodes=args.valid_unseen_episodes,
            eval_interval=args.eval_interval, max_steps=args.max_steps,
            device=device, save_path=args.save_path, load_path=args.load_path)

