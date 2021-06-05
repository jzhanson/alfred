import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

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
from models.utils.shared_optim import SharedRMSprop, SharedAdam

def setup_env(args):
    thor_env = ThorEnv()

    with open(os.path.join(os.environ['ALFRED_ROOT'], 'models/config/rewards.json'),
            'r') as jsonfile:
        reward_config = json.load(jsonfile)[args.reward_config_name]

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

    return ie

# Need to take gpu_id instead of device as argument because resnet needs gpu_id
def setup_model(args, gpu_id=None):
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

    if gpu_id is not None:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    action_embeddings = nn.Embedding(num_embeddings=num_actions,
            embedding_dim=args.action_embedding_dim).to(device)

    resnet_args = Namespace(visual_model='resnet', gpu=gpu_id is not None,
            gpu_index=gpu_id if gpu_id is not None else -1)
    # Even if args.use_visual_feature is False, still initialize a visual model
    # since it makes the code simpler and clearer, especially in awkward cases
    # surrounding args.separate_superpixel_model and args.superpixel_context
    # being 'visual'
    if 'resnet' in args.visual_model:
        visual_feature_size = 512
        visual_model = Resnet(resnet_args, use_conv_feat=False,
                pretrained=args.pretrained_visual_model,
                frozen=args.frozen_visual_model)
    else:
        print("visual model '" + args.visual_model + "' not supported")

    if args.separate_superpixel_model:
        if 'resnet' in args.superpixel_model:
            superpixel_feature_size = 512
            superpixel_model = Resnet(resnet_args, use_conv_feat=False,
                    pretrained=args.pretrained_visual_model,
                    frozen=args.frozen_visual_model)
        else:
            print("superpixel model '" + args.superpixel_model + "' not supported")
    else:
        superpixel_feature_size = visual_feature_size
        superpixel_model = visual_model

    if args.superpixel_fc_units is None:
        args.superpixel_fc_units = []
    elif type(args.superpixel_fc_units) is int:
        args.superpixel_fc_units = [args.superpixel_fc_units]
    # If there are superpixel fc layers, set superpixel_featre_size to the size
    # of the last layer
    if len(args.superpixel_fc_units) > 0:
        superpixel_feature_size = args.superpixel_fc_units[-1]
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

    visual_input_size = 0
    if args.use_visual_feature:
        visual_input_size += visual_feature_size
    if args.superpixel_context is not None:
        visual_input_size += superpixel_feature_size

    if args.fusion_model == 'SuperpixelFusion':
        prev_action_size = (args.action_embedding_dim +
            superpixel_feature_size)
    elif args.fusion_model == 'SuperpixelActionConcat':
        if args.superpixelactionconcat_add_superpixel_action:
            # args.action_embedding_dim should equal
            # superpixel_feature_size
            prev_action_size = args.action_embedding_dim
        else:
            prev_action_size = (args.action_embedding_dim +
                superpixel_feature_size)

    policy_model = LSTMPolicy(
            visual_feature_size=visual_input_size,
            prev_action_size=prev_action_size,
            lstm_hidden_size=args.lstm_hidden_dim, dropout=args.dropout,
            action_fc_units=args.action_fc_units,
            value_fc_units=args.value_fc_units,
            visual_fc_units=args.visual_fc_units,
            prev_action_after_lstm=args.prev_action_after_lstm,
            use_tanh=args.use_tanh).to(device)

    if args.fusion_model == 'SuperpixelFusion':
        model = SuperpixelFusion(action_embeddings=action_embeddings,
              visual_model=visual_model, superpixel_model=superpixel_model,
              policy_model=policy_model,
              use_visual_feature=args.use_visual_feature,
              superpixel_context=args.superpixel_context,
              slic_kwargs=slic_kwargs, boundary_pixels=args.boundary_pixels,
              neighbor_depth=args.neighbor_depth,
              neighbor_connectivity=args.neighbor_connectivity,
              black_outer=args.black_outer, device=device)
    elif args.fusion_model == 'SuperpixelActionConcat':
        model = SuperpixelActionConcat(action_embeddings=action_embeddings,
              visual_model=visual_model, superpixel_model=superpixel_model,
              policy_model=policy_model,
              use_visual_feature=args.use_visual_feature,
              superpixel_context=args.superpixel_context,
              slic_kwargs=slic_kwargs, boundary_pixels=args.boundary_pixels,
              neighbor_depth=args.neighbor_depth,
              neighbor_connectivity=args.neighbor_connectivity,
              black_outer=args.black_outer,
              single_interact=args.single_interact,
              zero_null_superpixel_features=args.zero_null_superpixel_features,
              navigation_superpixels=args.navigation_superpixels,
              add_superpixel_action=
              args.superpixelactionconcat_add_superpixel_action,
              device=device)

    try:
        model = model.to(device)
    except:
        model = model.to(device)

    if args.use_curiosity:
        if 'resnet' in args.curiosity_visual_encoder:
            curiosity_visual_encoder = Resnet(resnet_args, use_conv_feat=False,
                    pretrained=args.pretrained_visual_model,
                    frozen=args.frozen_visual_model)
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

        if args.fusion_model == 'SuperpixelFusion':
            # TODO: should curiosity use action embeddings or action/mask
            # logits for vanilla SuperpixelFusion?
            action_embedding_dim = (args.action_embedding_dim +
                superpixel_feature_size)
        elif args.fusion_model == 'SuperpixelActionConcat':
            if args.superpixelactionconcat_add_superpixel_action:
                action_embedding_dim = args.action_embedding_dim
            else:
                action_embedding_dim = (args.action_embedding_dim +
                    superpixel_feature_size)

        # TODO: make curiosity match use_visual_feature and superpixel_context
        curiosity_model = CuriosityIntrinsicReward(
                visual_encoder=curiosity_visual_encoder,
                action_embedding_dim=action_embedding_dim,
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

    return model, curiosity_model

def setup_optimizer(model, curiosity_model=None, optimizer_name='', lr=0.01,
        shared=False):
    # This set trick from
    # https://discuss.pytorch.org/t/how-to-train-several-network-at-the-same-time/4920/2
    # No need for a separate optimizer for curiosity - loss coefficients serve
    # the same purpose as a separate learning rate for the most part
    parameters = set(model.parameters())
    if curiosity_model is not None:
        parameters |= set(curiosity_model.parameters())
    if shared:
        # SharedSGD not implemented
        if optimizer_name == 'rmsprop':
            optimizer = SharedRMSprop(parameters, lr=lr)
        elif 'adam' in optimizer_name:
            amsgrad = 'amsgrad' in optimizer_name
            optimizer = SharedAdam(parameters, lr=lr, amsgrad=amsgrad)
        optimizer.share_memory()
    else:
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, lr=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=lr)
        elif 'adam' in optimizer_name:
            amsgrad = 'amsgrad' in optimizer_name
            optimizer = optim.Adam(parameters, lr=lr, amsgrad=amsgrad)
    return optimizer

def setup_train(rank, args, shared_model, shared_curiosity_model,
        shared_optimizer):
    # Set random seed for worker process
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if args.gpu_ids is not None:
        torch.cuda.manual_seed(args.seed + rank)

    ie = setup_env(args)

    if args.gpu_ids is not None:
        gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
        device = torch.device('cuda:' + str(gpu_id))
    else:
        gpu_id = None
        device = torch.device('cpu')

    model, curiosity_model = setup_model(args, gpu_id=gpu_id)

    if shared_optimizer is None:
        optimizer = setup_optimizer(shared_model,
                curiosity_model=shared_curiosity_model,
                optimizer_name=args.optimizer, lr=args.lr, shared=False)
    else:
        optimizer = shared_optimizer

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

    reset_kwargs = {
            'random_object_positions' : args.random_object_positions,
            'random_position' : args.random_position,
            'random_rotation' : args.random_rotation,
            'random_look_angle' : args.random_look_angle
    }

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
            navigation_superpixels=args.navigation_superpixels,
            curiosity_model=curiosity_model,
            curiosity_lambda=args.curiosity_lambda,
            seen_state_loss_coefficient=args.seen_state_loss_coefficient,
            scene_numbers=scene_numbers, reset_kwargs=reset_kwargs,
            max_trajectory_length=args.max_trajectory_length,
            frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            teacher_force=args.teacher_force, sample_action=args.sample_action,
            sample_mask=args.sample_mask, eval_interval=args.eval_interval,
            max_steps=args.max_steps, device=device, save_path=args.save_path,
            save_intermediate=args.save_intermediate,
            save_images_video=args.save_images_video,
            save_trajectory_info=args.save_trajectory_info,
            load_path=args.load_path)

if __name__ == '__main__':
    args = parse_args()

    # Set random seed for everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_ids is not None:
        torch.cuda.manual_seed(args.seed)

    if args.save_path is not None and not os.path.isdir(args.save_path):
        print('making directory', args.save_path)
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'args.json'), 'w') as jsonfile:
        json.dump(args.__dict__, jsonfile)
    trajectory_info_path = os.path.join(args.save_path, 'trajectory_info')
    if args.save_trajectory_info and not os.path.isdir(trajectory_info_path):
        print('making directory', trajectory_info_path)
        os.mkdir(trajectory_info_path)

    if args.gpu_ids is not None and type(args.gpu_ids) is int:
        args.gpu_ids = [args.gpu_ids]

    # Keep shared model on CPU
    shared_model, shared_curiosity_model = setup_model(args, gpu_id=None)

    if args.shared_optimizer:
        shared_optimizer = setup_optimizer(shared_model,
                curiosity_model=shared_curiosity_model,
                optimizer_name=args.optimizer, lr=args.lr, shared=True)
    else:
        shared_optimizer = None

    print('shared model parameters: ' + str(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad)))
    print('total parameters: ' + str(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad) * args.num_processes))

    # From https://github.com/dgriff777/rl_a3c_pytorch/blob/master/main.py,
    # which is from
    # https://github.com/pytorch/examples/tree/master/mnist_hogwild
    if args.gpu_ids is not None:
        mp.set_start_method('spawn')

    processes = []

    p = mp
    for rank in range(0, args.num_processes):
        p = mp.Process(target=setup_train, args=(rank, args, shared_model,
            shared_curiosity_model, shared_optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()
