import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import json
import os
import constants
import cv2
import numpy as np
import argparse
import time
import random
import gen.constants as constants
from utils.video_util import VideoSaver
from env.thor_env import ThorEnv
from env.interaction_exploration import InteractionExploration
from models.model.args import parse_args
from env.reward import InteractionReward
from models.nn.ie import SuperpixelFusion
import torch

if __name__ == '__main__':
    """
    Requires load-path, single-interact, slic arguments, use-gt-segmentation

    Can use reward_config_name, reward_rotations_look_angles,
    reward_state_changes, reward_persist_state, reward_repeat_discount,
    reward_use_novelty arguments.
    """
    args = parse_args()
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
            sample_contextual_action=False, use_masks=True)

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

    # Initialize a SuperpixelFusion class for the get_superpixel_masks_frame_crops
    sf = SuperpixelFusion(slic_kwargs=slic_kwargs,
            boundary_pixels=args.boundary_pixels,
            neighbor_depth=args.neighbor_depth,
            neighbor_connectivity=args.neighbor_connectivity,
            black_outer=args.black_outer)

    if args.single_interact:
        num_actions = len(constants.SIMPLE_ACTIONS)
        index_to_action = constants.INDEX_TO_ACTION_SIMPLE
    else:
        num_actions = len(constants.COMPLEX_ACTIONS)
        index_to_action = constants.INDEX_TO_ACTION_COMPLEX

    trajectory_jsonfiles = [fname for fname in os.listdir(args.load_path)
            if 'json' in fname]
    trajectory_jsonfiles.sort()
    #trajectory_jsonfiles = ['0.json']
    for trajectory_info_name in trajectory_jsonfiles:
        jsonfile_full_path = os.path.join(args.load_path, trajectory_info_name)

        with open(jsonfile_full_path, 'r') as jsonfile:
            trajectory_info = json.load(jsonfile)

        print(trajectory_info_name)
        print(trajectory_info.keys())
        frame = ie.reset(trajectory_info['scene_num'])
        event = ie.get_last_event()
        thor_env.restore_scene(trajectory_info['object_poses'], [], False)
        start_pose = trajectory_info['agent_pose_discrete']
        agent_height = event.metadata['agent']['position']['y']
        # This init_pose_action going around InteractionExploration might mess
        # up the initial position for reward calculation
        init_pose_action = {'action': 'TeleportFull',
                  'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                  'y': agent_height,
                  'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'rotation': start_pose[2] * 90,
                  'horizon': start_pose[3],
                  }
        event = thor_env.step(init_pose_action)

        for pred_action_index, pred_mask_index in zip(
                trajectory_info['pred_action_indexes'],
                trajectory_info['pred_mask_indexes']):
            # Make frame torch and (3, 300, 300) to match with code in
            # get_superpixel_masks_frame_crops or
            # get_gt_segmentation_masks_frame_crops
            if args.use_gt_segmentation:
                # Sometimes the environment's instance_segmentation_frame can
                # have different colors for objects (maybe color_to_object_id
                # also), especially if there are a lot of objects. However,
                # since SuperpixelFusion.get_gt_segmentation_masks_frame_crops
                # iterates through each pixel of the segmentation frame and
                # doesn't care about color_to_object_id, the results will be
                # the same even if instance_segmentation_frame has different
                # colors.
                masks, frame_crops = sf.get_gt_segmentation_masks_frame_crops(
                        torch.from_numpy(np.ascontiguousarray(event.frame).transpose(2,
                            0, 1)), event.instance_segmentation_frame)
            else:
                masks, frame_crops = sf.get_superpixel_masks_frame_crops(
                        torch.from_numpy(np.ascontiguousarray(event.frame).transpose(2, 0, 1)))

            selected_action = index_to_action[pred_action_index]
            selected_mask = (masks[pred_mask_index] if pred_mask_index >= 0
                    else None)
            _, reward, _, (action_success, event, err) = ie.step(
                    selected_action,
                    interact_mask=selected_mask)
            print(selected_action, action_success, reward, err)

