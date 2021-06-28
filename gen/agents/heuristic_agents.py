import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json
import constants
import cv2
import numpy as np
import argparse
import time
import random
import gen.constants as constants
from models.model.args import parse_args
from models.nn.ie import SuperpixelFusion
from models.train.train_rl_ie import check_thor, get_scene_numbers, setup_env
import torch
import multiprocessing as mp

def heuristic_rollout(ie, scene_num, single_interact=False,
        use_gt_segmentation=False, max_trajectory_length=None, slic_kwargs={},
        boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
        black_outer=False):
    if single_interact:
        num_actions = len(constants.SIMPLE_ACTIONS)
        index_to_action = constants.INDEX_TO_ACTION_SIMPLE
    else:
        num_actions = len(constants.COMPLEX_ACTIONS)
        index_to_action = constants.INDEX_TO_ACTION_COMPLEX

    frame = ie.reset(scene_num)
    event = ie.get_last_event()

    trajectory_info = {}
    trajectory_info['scene_num'] = scene_num
    trajectory_info['agent_pose_discrete'] = event.pose_discrete
    trajectory_info['object_poses'] = [{'objectName':
        obj['name'].split('(Clone)')[0], 'position': obj['position'],
        'rotation': obj['rotation']} for obj in event.metadata['objects']
        if obj['pickupable']]
    trajectory_info['pred_action_indexes'] = []
    trajectory_info['pred_mask_indexes'] = []
    trajectory_info['rewards'] = []
    eval_results = {}
    per_step_coverages = []
    eval_results['action_successes'] = []

    steps = 0
    done = False
    while not done and (max_trajectory_length is None or steps <
            max_trajectory_length):
        per_step_coverages.append(ie.get_coverages())
        # Make frame torch and (3, 300, 300) to match with code in
        # get_superpixel_masks_frame_crops or
        # get_gt_segmentation_masks_frame_crops
        if use_gt_segmentation:
            # Sometimes the environment's instance_segmentation_frame can
            # have different colors for objects (maybe color_to_object_id
            # also), especially if there are a lot of objects. However,
            # since SuperpixelFusion.get_gt_segmentation_masks_frame_crops
            # iterates through each pixel of the segmentation frame and
            # doesn't care about color_to_object_id, the results will be
            # the same even if instance_segmentation_frame has different
            # colors.
            masks, frame_crops = (SuperpixelFusion
                    .get_gt_segmentation_masks_frame_crops(
                        torch.from_numpy(np.ascontiguousarray(event.frame)
                            .transpose(2, 0, 1)),
                        event.instance_segmentation_frame,
                        boundary_pixels=boundary_pixels,
                        black_outer=black_outer))
        else:
            masks, frame_crops = (SuperpixelFusion
                    .get_superpixel_masks_frame_crops(
                        torch.from_numpy(np.ascontiguousarray(event.frame)
                            .transpose(2, 0, 1)), slic_kwargs=slic_kwargs,
                        boundary_pixels=boundary_pixels,
                        neighbor_depth=neighbor_depth,
                        neighbor_connectivity=neighbor_connectivity,
                        black_outer=black_outer))

        # TODO: select action and mask depending on random, random+, heuristic
        pred_action_index = random.randint(0, num_actions - 1)
        selected_action = index_to_action[pred_action_index]
        if selected_action not in constants.NAV_ACTIONS:
            pred_mask_index = random.randint(0, len(masks) - 1)
            selected_mask = masks[pred_mask_index]
        else:
            pred_mask_index = -1
            selected_mask = None
        _, reward, _, (action_success, event, err) = ie.step(
                selected_action,
                interact_mask=selected_mask)
        print(selected_action, action_success, reward, err)
        trajectory_info['pred_action_indexes'].append(pred_action_index)
        trajectory_info['pred_mask_indexes'].append(pred_mask_index)
        trajectory_info['rewards'].append(reward)
        eval_results['action_successes'].append(action_success)
        steps += 1


    eval_results['scene_name_or_num'] = scene_num
    eval_results['pred_action_indexes'] = (
            trajectory_info['pred_action_indexes'])
    eval_results['rewards'] = trajectory_info['rewards']
    # Get end-of-episode coverage results
    (navigation_coverage,
            navigation_poses_coverage,
            interaction_coverage_by_object,
            state_change_coverage_by_object,
            interaction_coverage_by_type,
            state_change_coverage_by_type) = ie.get_coverages()
    # coverage_* so the metrics are grouped together in tensorboard :P
    eval_results['coverage_navigation'] = navigation_coverage
    eval_results['coverage_navigation_pose'] = navigation_poses_coverage
    eval_results['coverage_interaction_by_object'] = (
            interaction_coverage_by_object)
    eval_results['coverage_state_change_by_object'] = (
            state_change_coverage_by_object)
    eval_results['coverage_interaction_by_type'] = (
            interaction_coverage_by_type)
    eval_results['coverage_state_change_by_type'] = (
            state_change_coverage_by_type)
    # Copy over per-step coverages
    for coverage_type, coverages in zip(constants.COVERAGE_TYPES,
            zip(*per_step_coverages)):
        k = 'per_step_coverage_' + coverage_type
        eval_results[k] = coverages

    return trajectory_info, eval_results

def setup_rollouts(rank, args, trajectory_sync):
    ie = setup_env(args)
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

    scene_numbers = get_scene_numbers(args.scene_numbers, args.scene_types)

    start_time = time.time()
    trajectory_local = 0
    total_steps = 0
    while True:
        # "Grab ticket" and increment train_steps_sync with the intention of
        # rolling out that trajectory
        with trajectory_sync.get_lock():
            trajectory_local = trajectory_sync.value
            trajectory_sync.value += 1
        if trajectory_local >= args.max_steps:
            break

        trajectory_info_save_path = os.path.join(args.save_path,
                'trajectory_info', str(trajectory_local) + '.json')
        eval_results_save_path = os.path.join(args.save_path,
                'eval_results', str(trajectory_local) + '.json')

        scene_num = scene_numbers[trajectory_local % len(scene_numbers)]

        trajectory_start_time = time.time()
        print('trajectory %d' % (trajectory_local))
        trajectory_info, eval_results = heuristic_rollout(ie, scene_num,
                single_interact=args.single_interact,
                use_gt_segmentation=args.use_gt_segmentation,
                max_trajectory_length=args.max_trajectory_length,
                slic_kwargs=slic_kwargs, boundary_pixels=args.boundary_pixels,
                neighbor_depth=args.neighbor_depth,
                neighbor_connectivity=args.neighbor_connectivity,
                black_outer=args.black_outer)

        if args.save_trajectory_info:
            with open(trajectory_info_save_path, 'w') as jsonfile:
                json.dump(trajectory_info, jsonfile, indent=0)
        with open(eval_results_save_path, 'w') as jsonfile:
            json.dump(eval_results, jsonfile, indent=0)

        total_steps += len(trajectory_info['rewards'])
        current_time = time.time()
        process_fps = total_steps / (current_time - start_time)
        process_trajectory_fps = len(trajectory_info['rewards']) / (
                current_time - trajectory_start_time)
        print('rank %d fps since start %.6f' % (rank, process_fps))
        print('rank %d trajectory fps %.6f' % (rank, process_trajectory_fps))

if __name__ == '__main__':
    args = parse_args()
    check_thor()
    # Set random seed for everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        os.mkdir(os.path.join(args.save_path, 'trajectory_info'))
        os.mkdir(os.path.join(args.save_path, 'eval_results'))

    mp.set_start_method('spawn')
    processes = []

    # Signed int should be large enough :P
    trajectory_sync = mp.Value('i', 0)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=setup_rollouts, args=(rank, args, trajectory_sync))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()
