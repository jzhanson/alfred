import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import gen.constants as constants
from models.model.args import parse_args
from models.model.rl_interaction import rollout_trajectory
from models.train.train_rl_ie import setup_env, setup_model

def evaluate(rank, shared_model, ie, evaluate_steps_sync, single_interact=False,
        use_masks=True, use_gt_segmentation=False,
        fusion_model='SuperpixelFusion', outer_product_sampling=False,
        inverse_score=False, zero_null_superpixel_features=True,
        navigation_superpixels=False, action_mask_score_combination='add',
        scene_numbers=None, reset_kwargs={}, max_trajectory_length=None,
        frame_stack=1, zero_fill_frame_stack=False, sample_action=True,
        sample_mask=True, max_steps=100, device=torch.device('cpu'),
        save_path=None):
    start_time = time.time()
    shared_model.eval()
    while True:
        trajectory_start_time = time.time()
        # "Grab ticket" with intention to rollout trajectory
        with evaluate_steps_sync.get_lock():
            evaluate_steps_local = evaluate_steps_sync.value
            evaluate_steps_sync.value += 1
        if evaluate_steps_local >= max_steps:
            break

        scene_num = scene_numbers[evaluate_steps_local % len(scene_numbers)]
        with torch.no_grad():
            trajectory_results = rollout_trajectory(ie, shared_model,
                    single_interact=single_interact, use_masks=use_masks,
                    use_gt_segmentation=use_gt_segmentation,
                    fusion_model=fusion_model,
                    outer_product_sampling=outer_product_sampling,
                    inverse_score=inverse_score,
                    zero_null_superpixel_features=zero_null_superpixel_features,
                    navigation_superpixels=navigation_superpixels,
                    action_mask_score_combination=action_mask_score_combination,
                    get_per_step_coverages=True,
                    max_trajectory_length=max_trajectory_length,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    sample_action=sample_action, sample_mask=sample_mask,
                    scene_name_or_num=scene_num, reset_kwargs=reset_kwargs,
                    verbose_rollouts=False, device=device)

        results_to_save = {}
        results_to_save['scene_name_or_num'] = (
                trajectory_results['scene_name_or_num'])
        results_to_save['action_successes'] = (
                trajectory_results['action_successes'])
        results_to_save['pred_action_indexes'] = (
                [pred_action_index.item() for pred_action_index in
                    trajectory_results['pred_action_indexes']])
        results_to_save['rewards'] = trajectory_results['rewards']
        # Get end-of-episode coverage results
        for coverage_type in constants.COVERAGE_TYPES:
            k = 'coverage_' + coverage_type
            results_to_save[k] = trajectory_results[k]
        per_step_coverages = trajectory_results['per_step_coverages']
        for coverage_type, coverages in zip(constants.COVERAGE_TYPES,
                zip(*per_step_coverages)):
            k = 'per_step_coverage_' + coverage_type
            results_to_save[k] = coverages
        print(results_to_save)

        results_save_path = os.path.join(save_path, str(evaluate_steps_local) +
                '.json')
        with open(results_save_path, 'w') as jsonfile:
            json.dump(results_to_save, jsonfile)

        current_time = time.time()

        if max_trajectory_length is not None:
            total_frames = evaluate_steps_local * max_trajectory_length
            total_fps = total_frames / (current_time - start_time)
            print('total FPS since start %.6f' % total_fps)
        trajectory_fps = len(trajectory_results['rewards']) / (current_time -
                trajectory_start_time)
        print('rank %d FPS over last trajectory %.6f' % (rank, trajectory_fps))

def setup_evaluate(rank, args, shared_model, evaluate_steps_sync):
    # Set random seed
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

    evaluate(rank, shared_model, ie, evaluate_steps_sync,
            single_interact=args.single_interact, use_masks=args.use_masks,
            use_gt_segmentation=args.use_gt_segmentation,
            fusion_model=args.fusion_model,
            outer_product_sampling=args.outer_product_sampling,
            inverse_score=args.inverse_score,
            zero_null_superpixel_features=args.zero_null_superpixel_features,
            navigation_superpixels=args.navigation_superpixels,
            action_mask_score_combination=args.action_mask_score_combination,
            scene_numbers=scene_numbers, reset_kwargs=reset_kwargs,
            max_trajectory_length=args.max_trajectory_length,
            frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            sample_action=args.sample_action, sample_mask=args.sample_mask,
            max_steps=args.max_steps, device=device, save_path=args.save_path)

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

    if args.gpu_ids is not None and type(args.gpu_ids) is int:
        args.gpu_ids = [args.gpu_ids]

    # One model per GPU
    #
    # Be careful about moving models carelessly between devices because the
    # custom Resnet class has a self.device member that won't be changed by
    # model.to(device)!
    shared_models = []
    if args.gpu_ids is not None:
        for gpu_id in args.gpu_ids:
            shared_model, _ = setup_model(args, gpu_id=gpu_id)
            shared_models.append(shared_model)
    else:
        shared_model, _ = setup_model(args, gpu_id=None)
        shared_models.append(shared_model)

    # TODO: load multiple model checkpoints and settings (from saved args
    # files) in a loop and allow graphing on the same axes
    for shared_model in shared_models:
        if args.load_path is not None:
            load_checkpoint(args.load_path, shared_model, None, None)

    print('shared model parameters: ' + str(sum(p.numel() for p in
        shared_models[0].parameters() if p.requires_grad)))
    print('total parameters: ' + str(sum(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad) for shared_model in
        shared_models)))

    # The rest of this file is from
    # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/main.py, which is
    # from https://github.com/pytorch/examples/tree/master/mnist_hogwild
    #
    # Don't know why dgriff777 only sets start method if running with gpu, code
    # hangs when running on CPU unless I set it
    mp.set_start_method('spawn')

    processes = []

    # Signed int should be large enough :P
    evaluate_steps_sync = mp.Value('i', 0)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=setup_evaluate, args=(rank, args,
            shared_models[rank % len(shared_models)], evaluate_steps_sync))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

    # List of (trajectory, step, coverage_type, coverage)
    pre_df = []
    # Graph each coverage type by step
    for trajectory_i in range(args.max_steps):
        trajectory_load_path = os.path.join(args.save_path, str(trajectory_i) + '.json')
        with open(trajectory_load_path, 'r') as jsonfile:
            trajectory_results = json.load(jsonfile)

        for coverage_type in constants.COVERAGE_TYPES:
            k = 'per_step_coverage_' + coverage_type
            for step_i in range(len(trajectory_results[k])):
                pre_df.append((trajectory_i, step_i,
                    coverage_type, trajectory_results[k][step_i]))
    df = pd.DataFrame(data=pre_df, columns=['trajectory', 'step',
        'coverage_type', 'coverage'])
    plt.clf()
    sns.lineplot(x='step', y='coverage', hue='coverage_type', estimator='mean',
            ci='sd', data=df)
    plt.savefig(os.path.join(args.save_path, 'coverage.png'))
