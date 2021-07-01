import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import math
import copy
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
from models.utils.helper_utils import superpixelactionconcat_index_to_action
import torch
import multiprocessing as mp

class RandomAgent(object):
    def __init__(self, single_interact=False, outer_product_sampling=False,
            navigation_superpixels=False):
        self.single_interact = single_interact
        self.outer_product_sampling = outer_product_sampling
        self.navigation_superpixels = navigation_superpixels
        if self.single_interact:
            self.actions = constants.SIMPLE_ACTIONS
            self.index_to_action = constants.INDEX_TO_ACTION_SIMPLE
        else:
            self.actions = constants.COMPLEX_ACTIONS
            self.index_to_action = constants.INDEX_TO_ACTION_COMPLEX

    def reset(self, ie):
        return

    def get_pred_action_mask_indexes(self, ie, masks,
            last_action_success=True):
        # Allow outer_product_sampling and navigation_superpixels for different
        # flavors of random action selection
        if self.outer_product_sampling:
            if args.navigation_superpixels:
                num_actions = len(self.actions) * len(masks)
            else:
                num_actions = (len(constants.NAV_ACTIONS) + (1 if
                    self.single_interact else len(constants.INT_ACTIONS)) *
                    len(masks))
            pred_action = (
                superpixelactionconcat_index_to_action(random.randint(0,
                num_actions - 1), num_actions,
                single_interact=self.single_interact,
                navigation_superpixels=self.navigation_superpixels))
            pred_action_index = self.actions.index(pred_action)
            if args.navigation_superpixels:
                pred_mask_index = pred_action_index % len(self.actions)
            else:
                # This will be nonsensical if a navigation action is chosen,
                # but that's okay because we set pred_mask_index to -1 at the
                # end of the function if a navigation action is chosen anyways
                pred_mask_index = (pred_action_index -
                        len(constants.NAV_ACTIONS)) % (1 if
                                self.single_interact else
                                len(constants.INT_ACTIONS))
        else:
            pred_action_index = random.randint(0, len(self.actions) - 1)
            pred_mask_index = random.randint(0, len(masks) - 1)

        if pred_action_index < len(constants.NAV_ACTIONS):
            pred_mask_index = -1
        return pred_action_index, pred_mask_index

class WallAgent(RandomAgent):
    """Navigates to closest wall, then walks the perimeter of the scene.
    """

    def __init__(self, **kwargs):
        super(WallAgent, self).__init__(**kwargs)

    def reset(self, ie):
        self.all_points = WallAgent.find_wall_points(ie.graph.points)
        self.reset_points(ie.env.last_event.pose_discrete)
        # self.path_to_wall will always be one longer than self.actions_to_wall
        # and contain both the current pose and the goal pose
        self.actions_to_destination, self.path_to_destination = (
                WallAgent.get_closest_actions_path(ie, self.points))

    def get_pred_action_mask_indexes(self, ie, masks,
            last_action_success=True):
        if not last_action_success:
            # Mark next location (i.e. would-be current location) as impossible
            # and replan
            print('Could not reach location ' +
                    str(self.path_to_destination[0]) +
                    ', marking as impossible')
            ie.graph.add_impossible_spot(self.path_to_destination[0])
            self.actions_to_destination, self.path_to_destination = (
                    WallAgent.get_closest_actions_path(ie, self.points))
        if len(self.actions_to_destination) == 0:
            if len(self.path_to_destination) > 1:
                # Didn't reach the target wall location
                print('Could not reach goal ' +
                        str(self.path_to_destination[-1]) +
                        ', marking as impossible')
                ie.graph.add_impossible_spot(self.path_to_destination[-1])
            self.points.remove(self.path_to_destination[-1][:2])
            if len(self.points) == 0:
                # Perimeter lap complete - reset self.points and "start over"
                self.reset_points(ie.env.last_event.pose_discrete)
            self.actions_to_destination, self.path_to_destination = (
                    WallAgent.get_closest_actions_path(ie, self.points))
        action = self.actions_to_destination[0]['action']
        self.actions_to_destination = self.actions_to_destination[1:]
        self.path_to_destination = self.path_to_destination[1:]
        pred_action_index = self.actions.index(action)
        return pred_action_index, -1

    def reset_points(self, pose_discrete):
        self.points = copy.deepcopy(self.all_points)
        if pose_discrete[:2] in self.points:
            self.points.remove(pose_discrete[:2])

    @classmethod
    def find_wall_points(cls, points):
        # A wall is a point where at least one cardinal direction is blocked (i.e.
        # not in points)
        tuple_points = [tuple(point) for point in list(points)]
        wall_points = []
        for x, z in tuple_points:
            has_direction_blocked = not all([(x-1, z) in tuple_points, (x+1, z) in
                tuple_points, (x, z-1) in tuple_points, (x, z+1) in tuple_points])
            if has_direction_blocked:
                wall_points.append((x, z))
        return wall_points

    @classmethod
    def get_closest_actions_path(cls, ie, points):
        current_x, current_z, current_rotation, current_look_angle = (
                ie.env.last_event.pose_discrete)
        raw_distances_to_points = [math.sqrt((current_x - x)**2 +
            (current_z - z)**2) for x, z in points]
        min_distance = min(raw_distances_to_points)
        min_indexes = [i for i, distance in
                enumerate(raw_distances_to_points) if distance ==
                min_distance]
        min_points = [points[min_index] for min_index in min_indexes]
        actions_paths = []
        for x, z in min_points:
            # Check all rotations for minimum action distance
            point_actions_paths = []
            for rotation in range(4):
                actions, path = ie.graph.get_shortest_path(
                        ie.env.last_event.pose_discrete, (x, z, rotation,
                            current_look_angle))
                point_actions_paths.append((actions, path))
            actions_paths.append(min(point_actions_paths, key=lambda ap:
                len(ap[0])))
        actions, path = min(actions_paths, key=lambda ap: len(ap[0]))
        return actions, path


def heuristic_rollout(ie, scene_num, agent, single_interact=False,
        use_gt_segmentation=False, max_trajectory_length=None, slic_kwargs={},
        boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
        black_outer=False):
    if single_interact:
        index_to_action = constants.INDEX_TO_ACTION_SIMPLE
    else:
        index_to_action = constants.INDEX_TO_ACTION_COMPLEX

    frame = ie.reset(scene_num)
    event = ie.get_last_event()
    agent.reset(ie)

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
    action_success = True
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
        pred_action_index, pred_mask_index = (agent
                .get_pred_action_mask_indexes(ie, masks,
                    last_action_success=action_success))
        selected_action = index_to_action[pred_action_index]
        selected_mask = (masks[pred_mask_index] if pred_mask_index >= 0 else
                None)
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

    if args.heuristic_agent == 'random':
        agent = RandomAgent(single_interact=args.single_interact)
    elif args.heuristic_agent == 'wall':
        agent = WallAgent(single_interact=args.single_interact)

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
        trajectory_info, eval_results = heuristic_rollout(ie, scene_num, agent,
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
