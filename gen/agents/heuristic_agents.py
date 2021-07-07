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
from env.find_one import crow_distance, get_end_poses_point_indexes
import torch
import multiprocessing as mp

def trajectory_index_break_tie(tiebreaker, l):
    trajectory_index, total_trajectories = tiebreaker
    chosen_index = (trajectory_index // total_trajectories) % len(l)
    if len(l) > 1:
        print('tiebreaker', chosen_index, len(l))
    return l[chosen_index]

def in_closed_receptacle(event, obj):
    if obj['parentReceptacles'] is not None:
        for parent_receptacle_id in obj['parentReceptacles']:
            parent_receptacle = event.get_object(parent_receptacle_id)
            if (parent_receptacle['openable'] and not
                    parent_receptacle['isOpen']):
                return True
    return False

def get_closed_parent_receptacle_ids(event, obj):
    closed_parent_receptacle_ids = []
    if obj['parentReceptacles'] is not None:
        for parent_receptacle_id in obj['parentReceptacles']:
            parent_receptacle = event.get_object(parent_receptacle_id)
            if (parent_receptacle['openable'] and not
                    parent_receptacle['isOpen']):
                closed_parent_receptacle_ids.append(parent_receptacle_id)
    return closed_parent_receptacle_ids

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

    def reset(self, ie, tiebreaker, scene_num):
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

class NavCoverageAgent(RandomAgent):
    """Navigates to closest wall, then walks the perimeter of the scene.
    """

    def __init__(self, **kwargs):
        super(NavCoverageAgent, self).__init__(**kwargs)

    def reset(self, ie, tiebreaker, scene_num):
        self.all_points = [tuple(point) for point in list(ie.graph.points)]
        self.reset_points(ie.env.last_event.pose_discrete)
        self.tiebreaker = tiebreaker
        self.scene_num = scene_num
        (self.actions_to_destination, self.path_to_destination,
                self.end_pose) = (self.get_closest_actions_path(ie))

    def get_pred_action_mask_indexes(self, ie, masks,
            last_action_success=True):
        if last_action_success:
            current_point = self.path_to_destination[0][:2]
            assert current_point == ie.env.last_event.pose_discrete[:2]
            if current_point in self.points:
                self.points.remove(current_point)
        else:
            # Mark next location (i.e. would-be current location) as impossible
            # and replan
            impossible_next_pose = self.path_to_destination[0]
            print('Could not reach location ' +
                    str(impossible_next_pose) +
                    ', marking as impossible')
            ie.graph.add_impossible_spot(impossible_next_pose)
            if impossible_next_pose[:2] in self.points:
                self.points.remove(impossible_next_pose[:2])
            (self.actions_to_destination, self.path_to_destination,
                    self.end_pose) = (self.get_closest_actions_path(ie))
        if len(self.actions_to_destination) == 0:
            if self.path_to_destination[0] != self.end_pose[:3]:
                # Didn't reach the target wall location
                print('Could not reach goal ' + str(self.end_pose) +
                    ', marking as impossible')
                ie.graph.add_impossible_spot(self.end_pose)
            end_point = self.path_to_destination[-1][:2]
            # Sometimes end point not in points due to having already been to
            # that location
            if end_point in self.points:
                self.points.remove(end_point)
            if len(self.points) == 0:
                # Perimeter lap complete - reset self.points and "start over"
                self.reset_points(ie.env.last_event.pose_discrete)
            (self.actions_to_destination, self.path_to_destination,
                    self.end_pose) = (self.get_closest_actions_path(ie))
        action = self.actions_to_destination[0]['action']
        self.actions_to_destination = self.actions_to_destination[1:]
        self.path_to_destination = self.path_to_destination[1:]
        pred_action_index = self.actions.index(action)
        return pred_action_index, -1

    def reset_points(self, pose_discrete):
        self.points = copy.deepcopy(self.all_points)
        if pose_discrete[:2] in self.points:
            self.points.remove(pose_discrete[:2])

    def get_closest_actions_path(self, ie):
        current_x, current_z, current_rotation, current_look_angle = (
                ie.env.last_event.pose_discrete)
        # Keep replanning until no impossible points are encountered
        need_restart = True
        impossible_point = None
        while need_restart:
            end_poses_actions_paths = []
            # Remove impossible points here so we're not removing inside the
            # same iteration loop
            if impossible_point is not None:
                # assert impossible_point in points
                self.points.remove(impossible_point)
                impossible_point = None
            need_restart = False
            for x, z in self.points:
                point_end_poses_actions_paths = []
                # Check all rotations of all given points for minimum action
                # distance - doesn't seem to noticably slow down the rollouts
                for rotation in range(4):
                    actions, path = ie.graph.get_shortest_path(
                            ie.env.last_event.pose_discrete, (x, z, rotation,
                                current_look_angle))
                    point_end_poses_actions_paths.append(((x, z, rotation,
                        current_look_angle), actions, path))
                min_end_pose, min_actions, min_path = min(
                        point_end_poses_actions_paths, key=lambda ap:
                        len(ap[0]))
                # Impossible spot will have undefined behavior
                if min_path[-1][:2] != (x, z):
                    # Rotation, look angle doesn't matter
                    ie.graph.add_impossible_spot((x, z, 0, 0))
                    impossible_point = (x, z)
                    need_restart = True
                    break
                else:
                    end_poses_actions_paths.append((min_end_pose, min_actions,
                        min_path))
        min_actions_distance = min([len(actions) for (end_pose, actions, path)
            in end_poses_actions_paths])
        # Current pose shouldn't appear in points, and no points should be
        # impossible
        assert min_actions_distance > 0
        min_end_poses_actions_paths = [(end_pose, actions, path) for (end_pose,
            actions, path) in end_poses_actions_paths if len(actions) ==
            min_actions_distance]
        end_pose, actions, path = trajectory_index_break_tie(self.tiebreaker,
                min_end_poses_actions_paths)
        return actions, path, end_pose

class WallAgent(NavCoverageAgent):
    def __init__(self, **kwargs):
        super(WallAgent, self).__init__(**kwargs)

    def reset(self, ie, tiebreaker, scene_num):
        self.all_points = WallAgent.find_wall_points(ie.graph.points)
        self.reset_points(ie.env.last_event.pose_discrete)
        # self.path_to_wall will always be one longer than self.actions_to_wall
        # and contain both the current pose and the goal pose
        self.tiebreaker = tiebreaker
        self.scene_num = scene_num
        (self.actions_to_destination, self.path_to_destination,
                self.end_pose) = (self.get_closest_actions_path(ie))

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

class IntCoverageAgent(RandomAgent):
    def __init__(self, **kwargs):
        super(IntCoverageAgent, self).__init__(**kwargs)

    def reset(self, ie, tiebreaker, scene_num):
        self.tiebreaker = tiebreaker
        self.scene_num = scene_num
        event = ie.env.last_event

        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        self.interactable_instance_ids = ie.env.prune_by_any_interaction(
                instance_ids)
        self.seen_interactions = set() # (instance_id, action)
        (self.closest_instance_id, self.closest_instance_action,
                self.actions_to_destination, self.path_to_destination,
                self.end_pose) = (
                        self.get_closest_interactable_instance_id_action(ie,
                        crow_distance=False))
        self.last_action_type = 'navigation'

    # A good amount of this code is taken from env/find_one.py
    def get_closest_interactable_instance_id_action(self, ie,
            crow_distance=False):
        """By crow distance, filtering by current interactability."""
        event = ie.env.last_event
        interactable_objects = [event.get_object(instance_id) for
                instance_id in self.interactable_instance_ids]
        current_interactable_objects_actions = []
        blocked_objects_actions = []
        for obj in interactable_objects:
            obj_accessible = True
            obj_seen_actions = [action for (target_instance_id, action) in
                    self.seen_interactions if target_instance_id ==
                    obj['objectId']]

            # Inventory object is inaccessible
            if ((len(event.metadata['inventoryObjects']) > 0 and
                obj['objectId'] ==
                event.metadata['inventoryObjects'][0]['objectId']) or
                in_closed_receptacle(event, obj)):
                obj_accessible = False

            if obj['pickupable'] and 'PickupObject' not in obj_seen_actions:
                if (len(event.metadata['inventoryObjects']) == 0 and
                        obj_accessible):
                    current_interactable_objects_actions.append((obj,
                        'PickupObject'))
                else:
                    blocked_objects_actions.append((obj, 'PickupObject'))
            if (obj['receptacle'] and 'PutObject' not in obj_seen_actions):
                # Sink is not in VAL_RECEPTACLE_OBJECTS, wait for SinkBasin
                # instead
                if (len(event.metadata['inventoryObjects']) > 0 and
                        obj['objectType'] != 'Sink' and
                        event.metadata['inventoryObjects'][0]['objectType'] in
                        constants.VAL_RECEPTACLE_OBJECTS[obj['objectType']] and
                        obj_accessible):
                    current_interactable_objects_actions.append((obj,
                        'PutObject'))
                else:
                    blocked_objects_actions.append((obj, 'PutObject'))
            if obj['openable']:
                if not obj['isOpen'] and 'OpenObject' not in obj_seen_actions:
                    if obj_accessible:
                        current_interactable_objects_actions.append((obj,
                            'OpenObject'))
                    else:
                        blocked_objects_actions.append((obj, 'OpenObject'))
                elif obj['isOpen'] and 'CloseObject' not in obj_seen_actions:
                    if obj_accessible:
                        current_interactable_objects_actions.append((obj,
                            'CloseObject'))
                    else:
                        blocked_objects_actions.append((obj, 'CloseObject'))
            if obj['toggleable']:
                if (not obj['isToggled'] and 'ToggleObjectOn' not in
                        obj_seen_actions):
                    if obj_accessible:
                        current_interactable_objects_actions.append((obj,
                            'ToggleObjectOn'))
                    else:
                        blocked_objects_actions.append((obj, 'ToggleObjectOn'))
                elif (obj['isToggled'] and 'ToggleObjectOff' not in
                        obj_seen_actions):
                    if obj_accessible:
                        current_interactable_objects_actions.append((obj,
                            'ToggleObjectOff'))
                    else:
                        blocked_objects_actions.append((obj, 'ToggleObjectOff'))
            if obj['sliceable'] and 'SliceObject' not in obj_seen_actions:
                if (len(event.metadata['inventoryObjects']) > 0 and 'Knife'
                        in event.metadata['inventoryObjects'][0][
                            'objectType'] and obj_accessible):
                    current_interactable_objects_actions.append((obj,
                        'SliceObject'))
                else:
                    blocked_objects_actions.append((obj, 'SliceObject'))

        # If current_interactable_objects_actions is empty, that means that
        # self.seen_interactions has not seen all possible interactions yet but
        # something is blocking the last few interactions, so fill
        # current_interactable_objects_actions with any action to unblock
        # TODO: it might be easier to just reset self.seen_interactions
        if len(current_interactable_objects_actions) == 0:
            assert len(blocked_objects_actions) > 0
            for obj, action in blocked_objects_actions:
                # For branches where an object is required but that object
                # could be in a closed receptacle, opening closed receptacles
                # that contain instances of those objects and objects outside
                # closed receptacles are equally weighted (i.e.  chosen by
                # distance to agent)

                # Object is in closed receptacle
                if in_closed_receptacle(event, obj):
                    # Open at least one closed receptacle - we might choose, in
                    # a pathological case, a closed receptacle inside another
                    # closed receptacle, but at that point, there's not much we
                    # can do
                    closed_parent_receptacle_ids = (
                            get_closed_parent_receptacle_ids(event, obj))
                    current_interactable_objects_actions.extend([
                            (event.get_object(parent_receptacle_id),
                                'OpenObject') for parent_receptacle_id in
                            closed_parent_receptacle_ids])
                # Holding uninteracted object (e.g. Box is pickupable and
                # openable)
                # Holding an object so can't pick up an uninteracted object
                # Holding an object so can't pick up knife for slicing
                # Holding the wrong object (not an object necessary to interact
                # with receptacle)
                elif (len(event.metadata['inventoryObjects']) > 0 and
                        (obj['objectId'] ==
                            event.metadata['inventoryObjects'][0]['objectId']
                            or action == 'PickupObject' or action ==
                            'SliceObject' or action == 'PutObject')):
                    # Find valid receptacle to put current object down
                    for possible_receptacle_obj in interactable_objects:
                        if (possible_receptacle_obj['receptacle'] and
                                event.metadata['inventoryObjects'][0][
                                    'objectType'] in
                                constants.VAL_RECEPTACLE_OBJECTS[
                                    obj['objectType']]):
                            if in_closed_receptacle(event,
                                    possible_receptacle_obj):
                                closed_parent_receptacle_ids = (
                                        get_closed_parent_receptacle_ids(event,
                                            obj))
                                current_interactable_objects_actions.extend([
                                    (event.get_object(parent_receptacle_id),
                                        'OpenObject') for parent_receptacle_id
                                    in closed_parent_receptacle_ids])
                            else:
                                current_interactable_objects_actions.append((
                                    obj, 'PutObject'))
                # Not having a knife to slice with
                elif action == 'SliceObject':
                    for possible_knife_obj in interactable_objects:
                        if 'Knife' in possible_knife_obj['objectType']:
                            if in_closed_receptacle(event, possible_knife_obj):
                                closed_parent_receptacle_ids = (
                                        get_closed_parent_receptacle_ids(event,
                                            obj))
                                current_interactable_objects_actions.extend([
                                    (event.get_object(parent_receptacle_id),
                                        'OpenObject') for parent_receptacle_id
                                    in closed_parent_receptacle_ids])
                            else:
                                current_interactable_objects_actions.append((
                                    obj, 'PickupObject'))
                # Not holding an object necessary to interact with receptacle
                elif action  == 'PutObject':
                    for possible_inside_obj in interactable_objects:
                        if (possible_inside_obj['objectType'] in
                                constants.VAL_RECEPTACLE_OBJECTS[
                                    obj['objectType']]):
                            if in_closed_receptacle(event,
                                    possible_inside_obj):
                                closed_parent_receptacle_ids = (
                                        get_closed_parent_receptacle_ids(event,
                                            obj))
                                current_interactable_objects_actions.extend([
                                    (event.get_object(parent_receptacle_id),
                                        'OpenObject') for parent_receptacle_id
                                    in closed_parent_receptacle_ids])
                            else:
                                current_interactable_objects_actions.append((
                                    obj, 'PickupObject'))

        # get_end_poses_point_indexes and get_shortest_path will path to the
        # next closest point for objects where the closest point is not
        # reachable
        #
        # end_poses and end_point_indexes will never be empty, but
        # current_interactable_objects_actions could be. closest_obj and
        # action will be found as long as current_interactable_objects_actions
        # is not empty
        assert len(current_interactable_objects_actions) > 0
        if crow_distance:
            min_distance = min([obj['distance'] for (obj, action) in
                current_interactable_objects_actions])
            closest_obj_actions = [(obj, action) for (obj, action) in
                    current_interactable_objects_actions if obj['distance'] ==
                    min_distance]
            closest_obj, action = trajectory_index_break_tie(self.tiebreaker,
                    closest_obj_actions)
            end_poses, end_point_indexes = get_end_poses_point_indexes(ie.env,
                    ie.graph, [closest_obj], self.scene_num)
            end_pose = trajectory_index_break_tie(self.tiebreaker, end_poses)
            # ie.graph.get_shortest_path will do its best if the path is
            # impossible, and we should still give that path a try
            actions, path = ie.graph.get_shortest_path(
                    ie.env.last_event.pose_discrete, end_pose)
        else:
            # Action distance
            end_poses, end_point_indexes = get_end_poses_point_indexes(ie.env,
                    ie.graph, [obj for (obj, action) in
                        current_interactable_objects_actions], self.scene_num)
            obj_action_end_poses_actions_paths = []
            for (obj, action), end_pose in zip(
                    current_interactable_objects_actions, end_poses):
                obj_action_end_poses_actions_paths.append((obj, action,
                    end_pose, *ie.graph.get_shortest_path(
                        ie.env.last_event.pose_discrete, end_pose)))
            min_actions = min([len(actions) for (_, _, _, actions, _) in
                obj_action_end_poses_actions_paths])
            min_obj_action_end_poses_actions_paths = [(obj, action, end_pose,
                actions, path) for (obj, action, end_pose, actions, path) in
                obj_action_end_poses_actions_paths if len(actions) ==
                min_actions]
            closest_obj, action, end_pose, actions, path = (
                    trajectory_index_break_tie(self.tiebreaker,
                        min_obj_action_end_poses_actions_paths))
        return closest_obj['objectId'], action, actions, path, end_pose

    def get_pred_action_mask_indexes(self, ie, masks,
            last_action_success=True, debug=False):
        if not last_action_success and self.last_action_type == 'navigation':
            # Mark next location (i.e. would-be current location) as
            # impossible
            print('Could not reach location ' +
                    str(self.path_to_destination[0]) +
                    ', marking as impossible')
            ie.graph.add_impossible_spot(self.path_to_destination[0])
            # Replan
            (self.closest_instance_id, self.closest_instance_action,
                    self.actions_to_destination,
                    self.path_to_destination, self.end_pose) = (self
                            .get_closest_interactable_instance_id_action(ie,
                                crow_distance=False))
        elif self.last_action_type == 'interaction':
            # Mark tried interaction as seen if tried once, even if
            # impossible. Retrying from a different location is a bit too
            # complicated for now
            self.seen_interactions.add((self.closest_instance_id,
                self.closest_instance_action))

            # If all interactions seen, empty interacted objects and "start over"
            if (len(self.seen_interactions) ==
                    constants.SCENE_INTERACTION_COVERAGES_BY_OBJECT[
                        self.scene_num]):
                self.seen_interactions = set()
            # Replan
            (self.closest_instance_id, self.closest_instance_action,
                    self.actions_to_destination,
                    self.path_to_destination, self.end_pose) = (self
                            .get_closest_interactable_instance_id_action(ie,
                                crow_distance=False))

        if debug:
            print('closest_instance_id action end pose',
                    self.closest_instance_id, self.closest_instance_action,
                    self.end_pose)
            print('current pose end pose', ie.env.last_event.pose_discrete,
                    self.end_pose)
            print('actions', self.actions_to_destination)
            print('path', self.path_to_destination)

        if len(self.actions_to_destination) == 0:
            if self.path_to_destination[0] != self.end_pose[:3]:
                # Didn't reach the target location
                print('Could not reach goal ' + str(self.end_pose) +
                    ', marking as impossible')
                ie.graph.add_impossible_spot(self.end_pose)
                # Try interaction anyway
            # Select mask and return from this branch - if no mask found that
            # will interact with the target object, return -1
            pred_mask_index = -1
            for i, mask in enumerate(masks):
                if (ie.env.mask_to_target_instance_id(mask) ==
                        self.closest_instance_id):
                    pred_mask_index = i
                    break
            self.last_action_type = 'interaction'
            return (self.actions.index(self.closest_instance_action),
                    pred_mask_index)

        self.last_action_type = 'navigation'
        action = self.actions_to_destination[0]['action']
        self.actions_to_destination = self.actions_to_destination[1:]
        self.path_to_destination = self.path_to_destination[1:]
        pred_action_index = self.actions.index(action)
        return pred_action_index, -1

def heuristic_rollout(ie, reset_kwargs, agent, tiebreaker,
        starting_look_angle=None, single_interact=False,
        use_gt_segmentation=False, max_trajectory_length=None, slic_kwargs={},
        boundary_pixels=0, neighbor_depth=0, neighbor_connectivity=2,
        black_outer=False):
    if single_interact:
        index_to_action = constants.INDEX_TO_ACTION_SIMPLE
    else:
        index_to_action = constants.INDEX_TO_ACTION_COMPLEX

    frame = ie.reset(**reset_kwargs)
    event = ie.get_last_event()
    if starting_look_angle is not None:
        # Teleport agent to given starting look angle
        start_pose = event.pose_discrete
        agent_height = event.metadata['agent']['position']['y']
        # This init_pose_action going around InteractionExploration might mess
        # up the initial look angle for reward calculation for eval_results,
        # but only if reward_rotation_look_angle is true
        init_pose_action = {'action': 'TeleportFull',
                  'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                  'y': agent_height,
                  'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'rotation': start_pose[2] * 90,
                  'horizon': starting_look_angle,
                  }
        event = ie.env.step(init_pose_action)
    agent.reset(ie, tiebreaker, reset_kwargs['scene_name_or_num'])

    trajectory_info = {}
    trajectory_info['scene_num'] = reset_kwargs['scene_name_or_num']
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

        pred_action_index, pred_mask_index = (agent
                .get_pred_action_mask_indexes(ie, masks,
                    last_action_success=action_success))
        selected_action = index_to_action[pred_action_index]
        selected_mask = (masks[pred_mask_index] if pred_mask_index >= 0 else
                None)

        frame, reward, _, (action_success, event, err) = ie.step(
                selected_action,
                interact_mask=selected_mask)
        print(selected_action, action_success, reward, err)
        trajectory_info['pred_action_indexes'].append(pred_action_index)
        trajectory_info['pred_mask_indexes'].append(pred_mask_index)
        trajectory_info['rewards'].append(reward)
        eval_results['action_successes'].append(action_success)
        steps += 1


    eval_results['scene_name_or_num'] = reset_kwargs['scene_name_or_num']
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
    elif args.heuristic_agent == 'navcoverage':
        agent = NavCoverageAgent(single_interact=args.single_interact)
    elif args.heuristic_agent == 'wall':
        agent = WallAgent(single_interact=args.single_interact)
    elif args.heuristic_agent == 'intcoverage':
        agent = IntCoverageAgent(single_interact=args.single_interact)

    if type(args.heuristic_look_angles) is int:
        args.heuristic_look_angles = [args.heuristic_look_angles]

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
        if args.heuristic_look_angles is not None:
            starting_look_angle = args.heuristic_look_angles[(trajectory_local
                    // len(scene_numbers)) % len(args.heuristic_look_angles)]
        else:
            starting_look_angle = None

        trajectory_start_time = time.time()
        print('trajectory %d' % (trajectory_local))
        reset_kwargs = {
                'scene_name_or_num' : scene_num,
                'random_object_positions' : args.random_object_positions,
                'random_position' : args.random_position,
                'random_rotation' : args.random_rotation,
                'random_look_angle' : args.random_look_angle,
        }
        tiebreaker = (trajectory_local, len(scene_numbers) * (1 if
            args.heuristic_look_angles is None else
            len(args.heuristic_look_angles)))
        trajectory_info, eval_results = heuristic_rollout(ie, reset_kwargs,
                agent, tiebreaker, starting_look_angle=starting_look_angle,
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
    trajectory_info_path = os.path.join(args.save_path, 'trajectory_info')
    eval_results_path = os.path.join(args.save_path, 'eval_results')
    if not os.path.isdir(trajectory_info_path):
        os.makedirs(trajectory_info_path)
    if not os.path.isdir(eval_results_path):
        os.makedirs(eval_results_path)

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
