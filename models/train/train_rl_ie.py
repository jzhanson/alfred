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
import time
from itertools import chain

import gen.constants as constants
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
from env.thor_env import ThorEnv
from env.interaction_exploration import InteractionExploration
from env.reward import InteractionReward
from models.model.args import parse_args
from models.nn.resnet import Resnet
from models.nn.ie import (CuriosityIntrinsicReward, LSTMPolicy,
        ResnetWrapper, SuperpixelFusion, SuperpixelActionConcat)
from models.model.rl_interaction import load_checkpoint, load_optimizer, train
from models.utils.shared_optim import SharedRMSprop, SharedAdam

def check_thor():
    # Load ai2thor because it will crash if downloaded for the first time on
    # multiple threads. Copied over from scripts/check_thor.py
    from ai2thor.controller import Controller
    c = Controller()
    c.start()
    event = c.step(dict(action="MoveAhead"))
    assert event.frame.shape == (300, 300, 3)
    print(event.frame.shape)
    print("Everything works!!!")

def get_scene_numbers(scene_numbers, scene_types, include_train=True,
        include_valid=False, include_test=False):
    """Contains logic for getting list of scene numbers from args.scene_numbers
    and args.scene_types.
    """
    output_scene_numbers = []
    if scene_numbers is not None:
        if type(scene_numbers) is int:
            output_scene_numbers = [scene_numbers]
        else:
            output_scene_numbers = scene_numbers
    else:
        # "'str' in" pattern will work for both list and single string
        if 'kitchen' in scene_types:
            if include_train:
                output_scene_numbers.extend(
                        constants.KITCHEN_TRAIN_SCENE_NUMBERS)
            if include_valid:
                output_scene_numbers.extend(
                        constants.KITCHEN_VALID_SCENE_NUMBERS)
            if include_test:
                output_scene_numbers.extend(
                        constants.KITCHEN_TEST_SCENE_NUMBERS)
        if 'living_room' in scene_types:
            if include_train:
                output_scene_numbers.extend(
                        constants.LIVING_ROOM_TRAIN_SCENE_NUMBERS)
            if include_valid:
                output_scene_numbers.extend(
                        constants.LIVING_ROOM_VALID_SCENE_NUMBERS)
            if include_test:
                output_scene_numbers.extend(
                        constants.LIVING_ROOM_TEST_SCENE_NUMBERS)
        if 'bedroom' in scene_types:
            if include_train:
                output_scene_numbers.extend(
                        constants.BEDROOM_TRAIN_SCENE_NUMBERS)
            if include_valid:
                output_scene_numbers.extend(
                        constants.BEDROOM_VALID_SCENE_NUMBERS)
            if include_test:
                output_scene_numbers.extend(
                        constants.BEDROOM_TEST_SCENE_NUMBERS)
        if 'bathroom' in scene_types:
            if include_train:
                output_scene_numbers.extend(
                        constants.BATHROOM_TRAIN_SCENE_NUMBERS)
            if include_valid:
                output_scene_numbers.extend(
                        constants.BATHROOM_VALID_SCENE_NUMBERS)
            if include_test:
                output_scene_numbers.extend(
                        constants.BATHROOM_TEST_SCENE_NUMBERS)
    return output_scene_numbers

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
        visual_feature_size = (512 if args.pretrained_visual_model else
                args.custom_resnet_planes[3])
        visual_model = Resnet(resnet_args, use_conv_feat=False,
                pretrained=args.pretrained_visual_model,
                frozen=args.frozen_visual_model,
                layers=args.custom_resnet_layers,
                inplanes=args.custom_resnet_inplanes,
                planes=args.custom_resnet_planes)
    else:
        print("visual model '" + args.visual_model + "' not supported")

    if args.separate_superpixel_model:
        if 'resnet' in args.superpixel_model:
            superpixel_feature_size = (512 if args.pretrained_visual_model else
                    args.custom_resnet_planes[3])
            # TODO: add a separate option for using a separate superpixel model
            # but a different custom ResNet architecture
            superpixel_model = Resnet(resnet_args, use_conv_feat=False,
                    pretrained=args.pretrained_visual_model,
                    frozen=args.frozen_visual_model,
                    layers=args.custom_resnet_layers,
                    inplanes=args.custom_resnet_inplanes,
                    planes=args.custom_resnet_planes)
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
    superpixel_model = ResnetWrapper(resnet_model=superpixel_model,
            fc_units=args.superpixel_fc_units, dropout=args.dropout,
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
                    frozen=args.frozen_visual_model,
                    layers=args.curiosity_resnet_layers,
                    inplanes=args.curiosity_resnet_inplanes,
                    planes=args.curiosity_resnet_planes)
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
    parameters = model.parameters()
    if curiosity_model is not None:
        # Have to use itertools.chain, since casting to set and |= messes up
        # the ordering of params in param_groups in the optimizer which is
        # important for saving and loading state dicts
        #
        # Discovered a bug where loading optimizer state dict led to parameters
        # and optimizer state having mismatched sizes because (1) Adam state
        # only includes averages for parameters that have grads, which means
        # that pretrained, frozen ResNet parameters don't have a saved
        # optimizer state and (2) when loading optimizer state dicts,
        # load_state_dict tries to sequentially match saved optimizer states to
        # parameters, which include the frozen ResNet parameters, so saved
        # optimizer states for trainable parameters e.g. policy_model were
        # being loaded into the wrong parameters
        parameters = chain(parameters, curiosity_model.parameters())
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
        shared_optimizer, train_steps_sync):
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

    # Here, we allow the model to be the same as the shared_model in case
    # we're only running a single worker so we don't have two copies of
    # models. I guess this could also be used for multiple workers with one
    # worker process's model being the shared model, but that kind of
    # defeats the purpose of having a shared model in the first place
    # (which is to allow the possibility of locking, and self-locking is
    # more complex and less clean)
    if args.num_processes > 1:
        model, curiosity_model = setup_model(args, gpu_id=gpu_id)
        # Load checkpoint but not optimizer
        if args.load_path is not None:
            load_checkpoint(args.load_path, model, curiosity_model, None)
    else:
        # Shared models will already be on the single GPU that is being used.
        # Checkpoint is already loaded
        model, curiosity_model = shared_model, shared_curiosity_model

    if shared_optimizer is None:
        optimizer = setup_optimizer(shared_model,
                curiosity_model=shared_curiosity_model,
                optimizer_name=args.optimizer, lr=args.lr, shared=False)
        if args.load_path is not None:
            # Do some filename tomfoolery to see if there are saved optimizer
            # files for the current worker process. If not (i.e. running with
            # more processes than before), load checkpoint's saved optimizer
            # state
            load_dir, checkpoint_name = os.path.split(args.load_path)
            if checkpoint_name == 'model.pth': # Wasn't using save_intermediate
                optimizer_checkpoint_name = 'optimizer_' + str(rank) + '.pth'
            else:
                # Choose the optimizer closest to the checkpoint for that worker
                load_step = int(checkpoint_name.split('.')[0].split('_')[1])
                optimizer_steps = [int(fname.split('.')[0].split('_')[1]) for
                        fname in os.listdir(load_dir) if 'optimizer' in fname
                        and int(fname.split('.')[0].split('_')[2]) == rank]
                if len(optimizer_steps) > 0:
                    closest_optimizer_step = min(optimizer_steps, key=lambda
                            x:abs(x - load_step))
                else:
                    closest_optimizer_step = -1
                optimizer_checkpoint_name = ('optimizer_' +
                        str(closest_optimizer_step) + '_' + str(rank) + '.pth')

            optimizer_load_path = os.path.join(load_dir,
                    optimizer_checkpoint_name)
            if os.path.isfile(optimizer_load_path):
                load_optimizer(optimizer_load_path, optimizer)
            else:
                load_optimizer(args.load_path, optimizer)
    else:
        # Likewise, shared optimizer state is already loaded
        optimizer = shared_optimizer

    scene_numbers = get_scene_numbers(args.scene_numbers, args.scene_types,
            include_train=args.include_train_scenes,
            include_valid=args.include_valid_scenes,
            include_test=args.include_test_scenes)

    reset_kwargs = {
            'random_object_positions' : args.random_object_positions,
            'random_position' : args.random_position,
            'random_rotation' : args.random_rotation,
            'random_look_angle' : args.random_look_angle
    }

    train(rank, args.num_processes, model, shared_model, ie, optimizer,
            train_steps_sync, gamma=args.gamma, tau=args.tau,
            policy_loss_coefficient=args.policy_loss_coefficient,
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
            action_mask_score_combination=args.action_mask_score_combination,
            curiosity_model=curiosity_model,
            shared_curiosity_model=shared_curiosity_model,
            curiosity_loss_coefficient=args.curiosity_loss_coefficient,
            seen_state_loss_coefficient=args.seen_state_loss_coefficient,
            scene_numbers=scene_numbers, reset_kwargs=reset_kwargs,
            max_trajectory_length=args.max_trajectory_length,
            frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            teacher_force=args.teacher_force, sample_action=args.sample_action,
            sample_mask=args.sample_mask, eval_interval=args.eval_interval,
            max_steps=args.max_steps, device=device,
            save_path=args.save_path, save_intermediate=args.save_intermediate,
            save_images_video=args.save_images_video,
            save_trajectory_info=args.save_trajectory_info,
            verbose_rollouts=args.verbose_rollouts)

if __name__ == '__main__':
    args = parse_args()

    check_thor()
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

    # Put shared models on GPU if there's only one process and we're using GPU
    # to save the CPU memory of a "useless" shared_model
    #
    # Be careful about moving models carelessly between devices because the
    # custom Resnet class has a self.device member that won't be changed by
    # model.to(device)!
    shared_model, shared_curiosity_model = setup_model(args,
            gpu_id=args.gpu_ids[0] if args.gpu_ids is not None and
            args.num_processes == 1 else None)

    if args.shared_optimizer:
        shared_optimizer = setup_optimizer(shared_model,
                curiosity_model=shared_curiosity_model,
                optimizer_name=args.optimizer, lr=args.lr, shared=True)
    else:
        shared_optimizer = None

    if args.load_path is not None:
        train_steps = load_checkpoint(args.load_path, shared_model,
                shared_curiosity_model, shared_optimizer)
    else:
        train_steps = 0

    print('shared model parameters: ' + str(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad)))
    print('total parameters: ' + str(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad) * args.num_processes))

    # The rest of this file is from
    # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/main.py, which is
    # from https://github.com/pytorch/examples/tree/master/mnist_hogwild
    #
    # Don't know why dgriff777 only sets start method if running with gpu, code
    # hangs when running on CPU unless I set it
    mp.set_start_method('spawn')

    processes = []

    # Signed int should be large enough :P
    train_steps_sync = mp.Value('i', train_steps)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=setup_train, args=(rank, args, shared_model,
            shared_curiosity_model, shared_optimizer, train_steps_sync))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

