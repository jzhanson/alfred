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
from gen.scripts.replay_trajectories import (get_object_counts_visible,
        get_object_counts_in_frame)
import torch
import multiprocessing as mp

# This is mostly copied from gen/scripts/replay_trajectories.py
def setup_generate(rank, args, lock, trajectories_properties,
        objects_properties):
    ie = setup_env(args)
    thor_env = ie.env

    scene_numbers = get_scene_numbers(args.scene_numbers, args.scene_types,
            include_train=args.include_train_scenes,
            include_valid=args.include_valid_scenes,
            include_test=args.include_test_scenes)

    start_time = time.time()
    total_steps = 0
    done = False
    while not done:
        scene_num = random.choice(scene_numbers)

        save_scene = args.create_scene_dataset
        save_object = args.create_object_dataset

        if args.starting_look_angles is not None:
            starting_look_angle = random.choice(args.starting_look_angles)
        else:
            starting_look_angle = None

        trajectory_start_time = time.time()
        frame = ie.reset(scene_num,
                random_object_positions=args.random_object_positions,
                random_position=args.random_position,
                random_rotation=args.random_rotation,
                random_look_angle=args.random_look_angle)
        event = ie.get_last_event()
        start_pose = event.pose_discrete
        if starting_look_angle is not None:
            agent_height = event.metadata['agent']['position']['y']
            # This init_pose_action going around InteractionExploration might mess
            # up the initial position for reward calculation
            init_pose_action = {'action': 'TeleportFull',
                      'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                      'y': agent_height,
                      'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                      'rotateOnTeleport': True,
                      'rotation': start_pose[2] * 90,
                      'horizon': starting_look_angle,
                      }
            event = thor_env.step(init_pose_action)
            start_pose = (*start_pose[:3], starting_look_angle)
        else:
            starting_look_angle = start_pose[3]

        if save_scene:
            scene_info = {}
            scene_info['scene_num'] = scene_num
            scene_info['start_pose'] = start_pose
            # object_counts_visible can't be computed post-hoc because we don't
            # have the environment to figure out whether objects are within
            # VISIBILITY_DISTANCE, but object_counts_in_frame can - we still
            # include it for faster dataset loading
            # It's also difficult to keep track of the distances of all
            # (visible) objects in the scene. It's probably easier and costs
            # much less memory to replay all trajectories to rebuild the
            # dataset with a different VISIBILITY_DISTANCE than it is to record
            # the object distances for every frame and filter objects by
            # distance in the dataloader.
            scene_info['object_counts_visible'] = []
            scene_info['object_counts_in_frame'] = []
            if args.save_segmentation:
                # Convert color_to_object_id to dict of have string keys since
                # json.dump doesn't like tuples
                color_to_object_id_str_keys = {}
                for k, v in event.color_to_object_id.items():
                    color_to_object_id_str_keys[str(k)] = v
                scene_info['color_to_object_id'] = color_to_object_id_str_keys
        if save_object:
            object_info = {}
            object_info['scene_num'] = scene_num
            object_info['start_pose'] = start_pose
            object_info['type'] = []
            object_info['object_counts_in_crop'] = []
            # Record the distance of each object, since it's not a lot of
            # memory and doable to iterate through all the images in the
            # dataset and only select the ones within a certain distance
            object_info['distance'] = []

        step = 0
        # Save frame and segmentation
        if args.high_res_images:
            image_extension = '.jpg'
        else:
            image_extension = '.png'
        if save_scene:
            # TODO: do we need to save superpixels? It would be faster but
            # for what purpose
            scene_info['object_counts_visible'].append(
                    get_object_counts_visible(event))
            scene_info['object_counts_in_frame'].append(
                    get_object_counts_in_frame(event))
        if save_object:
            frame_crops = []
            for object_id, (start_x, start_y, end_x, end_y) in (
                    event.instance_detections2D.items()):
                if start_x == end_x or start_y == end_y:
                    continue
                obj = event.get_object(object_id)
                if (obj is not None and obj['objectType'] not in
                        args.excluded_object_types):
                    max_y, min_y, max_x, min_x = (
                            SuperpixelFusion
                            .get_max_min_y_x_with_boundary(event.frame,
                                [start_y, end_y], [start_x, end_x],
                                args.boundary_pixels))
                    frame_crop = event.frame[min_y:max_y, min_x:max_x, :]
                    object_info['object_counts_in_crop'].append(
                            get_object_counts_in_frame(event, y_bounds=(min_y,
                                max_y), x_bounds=(min_x, max_x)))

                    if args.black_outer:
                        mask = np.sum(event.instance_segmentation_frame ==
                            event.object_id_to_color[object_id], 2) == 3
                        frame_crop = (SuperpixelFusion
                                .get_black_outer_frame_crops([frame_crop],
                                    [(min_y, max_y, min_x, max_x)],
                                    [mask])[0])
                    frame_crops.append(frame_crop)
                    object_info['type'].append(
                            constants.ALL_OBJECTS.index(obj['objectType']))
                    object_info['distance'].append(obj['distance'])

        total_steps += 1
        current_time = time.time()
        process_fps = total_steps / (current_time - start_time)
        process_trajectory_fps = 1 / (current_time - trajectory_start_time)
        print('rank %d fps since start %.6f' % (rank, process_fps))
        print('rank %d trajectory fps %.6f' % (rank, process_trajectory_fps))

        lock.acquire()
        save_this_trajectory = False
        try:
            seen_scene = scene_num in constants.TRAIN_SCENE_NUMBERS
            if save_scene:
                # Seen/unseen scene (True for seen, False for unseen), has
                # objects within outside visibility distance, look angle,
                # number of objects in frame greater than cluttered scene
                # threshold
                trajectory_properties = (seen_scene,
                        sum(scene_info['object_counts_visible'][0]) <
                        sum(scene_info['object_counts_in_frame'][0]),
                        starting_look_angle,
                        sum(scene_info['object_counts_in_frame'][0]) >
                            args.cluttered_scene_threshold)

                if (trajectories_properties.count(trajectory_properties) <
                        args.rejection_n):
                    save_this_trajectory = True
            if save_object:
                object_info_to_save = {}
                object_info_to_save['scene_num'] = scene_num
                object_info_to_save['start_pose'] = start_pose
                object_info_to_save['type'] = []
                object_info_to_save['object_counts_in_crop'] = []
                object_info_to_save['distance'] = []
                frame_crops_to_save = []
                # Seen/unseen scene (True for seen, False for unseen), object
                # is within visibility distance, look angle, number of objects
                # in frame greater than cluttered crop threshold, for each
                # object type
                for (object_type, distance, object_count_in_crop,
                        frame_crop) in zip(object_info['type'],
                                object_info['distance'],
                                object_info['object_counts_in_crop'],
                                frame_crops):
                    object_properties = (seen_scene, distance <
                            constants.VISIBILITY_DISTANCE, starting_look_angle,
                            sum(object_count_in_crop) >
                            args.cluttered_crop_threshold)
                    if objects_properties[object_type].count(
                            object_properties) < args.rejection_n:
                        save_this_trajectory = True
                        object_info_to_save['type'].append(object_type)
                        object_info_to_save['object_counts_in_crop'].append(
                                object_count_in_crop)
                        object_info_to_save['distance'].append(distance)
                        frame_crops_to_save.append(frame_crop)
                        # Need to reassign to dict value so manager will pick
                        # up the change - append doesn't work because it doesn't
                        # trigger manager to update its own state
                        # https://stackoverflow.com/a/46228938
                        objects_properties[object_type] += [object_properties]
                    # Append object_properties here to avoid repeating
                    # code+loop
                    elif (save_this_trajectory and not
                            args.balanced_object_dataset):
                        objects_properties[object_type].append(
                                object_properties)

            if save_this_trajectory:
                trajectory_index = len(trajectories_properties)
                trajectories_properties.append(trajectory_properties)
                print('writing trajectory', trajectory_index)
                print('trajectories_properties', trajectories_properties)
                print('objects_properties', objects_properties)

            # Stop when scenes are done (if constructing scene dataset) and
            # when objects are done (if constructing object dataset)
            scenes_complete = True
            objects_complete = True
            num_look_angles = (len(args.starting_look_angles) if
                    args.starting_look_angles is not None else
                    int((constants.HORIZON_BOUNDS[1] -
                        constants.HORIZON_BOUNDS[0]) /
                        constants.HORIZON_GRANULARITY))
            if save_scene:
                total_configurations = (2 * 2 * num_look_angles * 2 *
                        args.rejection_n)
                if len(trajectories_properties) < total_configurations:
                    scenes_complete = False
            if save_object:
                per_type_configurations = (2 * 2 * num_look_angles * 2 *
                        args.rejection_n)
                for object_type, object_properties in (
                        objects_properties.items()):
                    if (constants.ALL_OBJECTS[object_type] not in
                            args.excluded_object_types and
                            len(object_properties) < per_type_configurations):
                        objects_complete = False
                        break
            done = scenes_complete and objects_complete
        finally:
            lock.release()

        # Trajectories are saved both-or-nothing for scene and objects, if
        # you're constructing both datasets. If you want a "balanced" scene
        # dataset, don't construct an object dataset so save_this_trajectory
        # won't be "contaminated" by accounting for objects. If you want a
        # "balanced" object dataset, use args.balanced_object_dataset and it
        # doesn't matter whether you construct a scene dataset alongside it
        # because object saving under that argument only saves objects that are
        # new
        if save_this_trajectory:
            trajectory_save_path = os.path.join(args.save_path,
                    str(trajectory_index))
            if save_scene:
                scene_save_path = os.path.join(trajectory_save_path, 'scene')
                if not os.path.isdir(scene_save_path):
                    os.makedirs(scene_save_path)

                frame_save_path = os.path.join(scene_save_path, '%05d' % step
                        + image_extension)
                cv2.imwrite(frame_save_path,
                        cv2.cvtColor(event.frame.astype('uint8'),
                            cv2.COLOR_RGB2BGR))
                if args.save_segmentation:
                    segmentation_save_path = os.path.join(scene_save_path,
                            '%05d_segmentation' % step + image_extension)
                    cv2.imwrite(segmentation_save_path,
                            cv2.cvtColor(event.instance_segmentation_frame
                                .astype('uint8'), cv2.COLOR_RGB2BGR))
                scene_info_save_path = os.path.join(scene_save_path, 'info.json')
                with open(scene_info_save_path, 'w') as jsonfile:
                    json.dump(scene_info, jsonfile)
            if save_object:
                object_save_path = os.path.join(trajectory_save_path, 'object')
                if not os.path.isdir(object_save_path):
                    os.makedirs(object_save_path)
                if args.balanced_object_dataset:
                    object_info = object_info_to_save
                    frame_crops = frame_crops_to_save
                for i, frame_crop in enumerate(frame_crops):
                    crop_save_path = os.path.join(object_save_path, ('%05d' %
                        i) + image_extension)
                    cv2.imwrite(crop_save_path,
                            cv2.cvtColor(frame_crop.astype('uint8'),
                                cv2.COLOR_RGB2BGR))
                object_info_save_path = os.path.join(object_save_path, 'info.json')
                with open(object_info_save_path, 'w') as jsonfile:
                    json.dump(object_info, jsonfile)

if __name__ == '__main__':
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

    # In case only one object type was provided
    if type(args.excluded_object_types) is str:
        args.excluded_object_types = [args.excluded_object_types]

    mp.set_start_method('spawn')
    processes = []

    with mp.Manager() as manager:
        lock = manager.Lock()
        # No need to save trajectories_properties or objects_properties, since
        # those can be reconstructed from the scene/object infos for each
        # trajectory
        trajectories_properties = manager.list()
        objects_properties = manager.dict()
        for i in range(len(constants.ALL_OBJECTS)):
            # for python 3.6+, use manager.list() to automatically propogate
            # changes to lists inside a manager.dict()
            # https://stackoverflow.com/a/46228938
            objects_properties[i] = []
        for rank in range(0, args.num_processes):
            p = mp.Process(target=setup_generate, args=(rank, args,
                lock, trajectories_properties,
                objects_properties))
            p.start()
            processes.append(p)
            time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()

