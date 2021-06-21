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
from utils.video_util import VideoSaver
from models.model.args import parse_args
from models.nn.ie import SuperpixelFusion
from models.train.train_rl_ie import check_thor, setup_env
import torch
import multiprocessing as mp

def get_object_counts_visible(event):
    object_counts = [0 for _ in constants.ALL_OBJECTS]
    objects = event.metadata['objects']
    for obj in objects:
        object_type = obj['objectType']
        if obj['visible'] and object_type in constants.ALL_OBJECTS:
            object_index = constants.ALL_OBJECTS.index(object_type)
            object_counts[object_index] += 1
    return object_counts

def get_object_counts_in_frame(event):
    object_ids_in_frame = set()
    for i in range(event.instance_segmentation_frame.shape[0]):
        for j in range(event.instance_segmentation_frame.shape[1]):
            color = tuple(event.instance_segmentation_frame[i][j])
            object_ids_in_frame.add(event.color_to_object_id[color])
    object_counts = [0 for _ in constants.ALL_OBJECTS]
    for object_id in object_ids_in_frame:
        obj = event.get_object(object_id)
        if obj is None:
            # Some objects like "Cube", "Room", "Ceiling" aren't kept track of
            # by the thor environment
            continue
        object_type = obj['objectType']
        object_index = constants.ALL_OBJECTS.index(object_type)
        object_counts[object_index] += 1
    return object_counts

def setup_replay(rank, args, trajectory_jsonfiles, trajectory_index_sync):
    ie = setup_env(args)
    thor_env = ie.env
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
        index_to_action = constants.INDEX_TO_ACTION_SIMPLE
    else:
        num_actions = len(constants.COMPLEX_ACTIONS)
        index_to_action = constants.INDEX_TO_ACTION_COMPLEX

    start_time = time.time()
    total_steps = 0
    trajectory_index_local = None
    while True:
        # "Grab ticket" and increment train_steps_sync with the intention of
        # replaying out that trajectory
        with trajectory_index_sync.get_lock():
            if trajectory_index_local is None: # First iteration, even if loading
                trajectory_index_local = trajectory_index_sync.value
            else:
                trajectory_index_local = trajectory_index_sync.value
            trajectory_index_sync.value += 1
        if trajectory_index_local >= len(trajectory_jsonfiles):
            break

        trajectory_info_name = trajectory_jsonfiles[trajectory_index_local]
        jsonfile_full_path = os.path.join(args.load_path, trajectory_info_name)
        with open(jsonfile_full_path, 'r') as jsonfile:
            trajectory_info = json.load(jsonfile)

        trajectory_number = int(trajectory_info_name.split('.')[0])
        replay_save_path = os.path.join(args.save_path, str(trajectory_number))
        # Trajectory is already replayed (in a previous script run) if
        # trajectory_replay_dict is saved
        trajectory_replay_dict_save_path = os.path.join(replay_save_path,
                'info.json')
        if os.path.isdir(replay_save_path):
            if os.path.isfile(trajectory_replay_dict_save_path):
                continue
            # If directory exists but trajectory_replay_dict was not saved to
            # file (i.e. trajectory not completely replayed), no need to do
            # anything
        else:
            os.mkdir(replay_save_path)

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

        step = 0
        object_counts_visible = []
        object_counts_in_frame = []
        trajectory_replay_dict = {}
        if args.save_segmentation:
            # Convert color_to_object_id to dict of have string keys since
            # json.dump doesn't like tuples
            color_to_object_id_str_keys = {}
            for k, v in event.color_to_object_id.items():
                color_to_object_id_str_keys[str(k)] = v
            trajectory_replay_dict['color_to_object_id'] = color_to_object_id_str_keys
        trajectory_start_time = time.time()
        print('trajectory %d/%d ' % (trajectory_index_local,
            len(trajectory_jsonfiles)))
        for pred_action_index, pred_mask_index in zip(
                trajectory_info['pred_action_indexes'],
                trajectory_info['pred_mask_indexes']):
            # Save frame and segmentation
            if args.high_res_images:
                image_extension = '.jpg'
            else:
                image_extension = '.png'
            frame_save_path = os.path.join(replay_save_path, '%05d' % step +
                    image_extension)
            cv2.imwrite(frame_save_path,
                    cv2.cvtColor(event.frame.astype('uint8'),
                        cv2.COLOR_RGB2BGR))
            # TODO: also save bounding boxes for each segmentation so we don't
            # have to compute it in data loader?
            if args.save_segmentation:
                segmentation_save_path = os.path.join(replay_save_path,
                        '%05d_segmentation' % step + image_extension)
                cv2.imwrite(segmentation_save_path,
                        cv2.cvtColor(event.instance_segmentation_frame
                            .astype('uint8'), cv2.COLOR_RGB2BGR))

            # TODO: do we need to save superpixels? It would be faster but for
            # what purpose
            object_counts_visible.append(get_object_counts_visible(event))
            object_counts_in_frame.append(get_object_counts_in_frame(event))
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
                masks, frame_crops = (SuperpixelFusion
                        .get_gt_segmentation_masks_frame_crops(
                            torch.from_numpy(np.ascontiguousarray(event.frame)
                                .transpose(2, 0, 1)),
                            event.instance_segmentation_frame,
                            boundary_pixels=args.boundary_pixels,
                            black_outer=args.black_outer))
            else:
                masks, frame_crops = (SuperpixelFusion
                        .get_superpixel_masks_frame_crops(
                            torch.from_numpy(np.ascontiguousarray(event.frame)
                                .transpose(2, 0, 1)), slic_kwargs=slic_kwargs,
                            boundary_pixels=args.boundary_pixels,
                            neighbor_depth=args.neighbor_depth,
                            neighbor_connectivity=args.neighbor_connectivity,
                            black_outer=args.black_outer))

            selected_action = index_to_action[pred_action_index]
            print('masks, pred_mask_index', len(masks), pred_mask_index)
            selected_mask = (masks[pred_mask_index] if pred_mask_index >= 0
                    else None)
            _, reward, _, (action_success, event, err) = ie.step(
                    selected_action,
                    interact_mask=selected_mask)
            print(selected_action, action_success, reward, err)
            step += 1
        # object_counts_visible can't be computed post-hoc because we don't
        # have the environment to figure out whether objects are within
        # VISIBILITY_DISTANCE, but object_counts_in_frame can - we still
        # include it for faster dataset loading
        trajectory_replay_dict['object_counts_visible'] = object_counts_visible
        trajectory_replay_dict['object_counts_in_frame'] = (
                object_counts_in_frame)
        with open(trajectory_replay_dict_save_path, 'w') as jsonfile:
            json.dump(trajectory_replay_dict, jsonfile)

        total_steps += step
        current_time = time.time()
        process_fps = total_steps / (current_time - start_time)
        process_trajectory_fps = step / (current_time - trajectory_start_time)
        print('rank %d fps since start %.6f' % (rank, process_fps))
        print('rank %d trajectory fps %.6f' % (rank, process_trajectory_fps))

if __name__ == '__main__':
    """
    Requires load-path, single-interact, slic arguments, use-gt-segmentation

    Can use reward_config_name, reward_rotations_look_angles,
    reward_state_changes, reward_persist_state, reward_repeat_discount,
    reward_use_novelty arguments.
    """
    args = parse_args()
    check_thor()
    # Set random seed for everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TODO: access existing master index? Initialize master index? Do we need a
    # master index?
    if os.path.isdir(args.save_path):
        pass
    else:
        os.makedirs(args.save_path)

    trajectory_jsonfiles = [fname for fname in os.listdir(args.load_path)
            if 'json' in fname]
    trajectory_jsonfiles.sort()

    mp.set_start_method('spawn')
    processes = []

    # Signed int should be large enough :P
    trajectory_index_sync = mp.Value('i', 0)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=setup_replay, args=(rank, args,
            trajectory_jsonfiles, trajectory_index_sync))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()
