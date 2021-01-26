import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import random
import json
import numpy as np

import cv2
from env.thor_env import ThorEnv
import gen.constants as constants
from gen.graph.graph_obj import Graph

# Available scenes are [1, 30], [201, 230], [301, 330], and [401, 430]
# Tragically this is hardcoded in ai2thor 2.1.0 in
# ai2thor/controller.py line 429
AVAILABLE_SCENE_NUMBERS = list(range(1, 31)) + list(range(201, 231)) + \
    list(range(301, 331)) + list(range(401, 431)) # 20 scenes per room type

ACTIONS_DONE = 'Done'
ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown',
        'Done']
NUM_ACTIONS = len(ACTIONS)
INDEX_TO_ACTION = dict(enumerate(ACTIONS))
ACTION_TO_INDEX = dict((v,k) for k,v in INDEX_TO_ACTION.items())

# NOTE: these don't really matter yet
# TODO: agent can cheese reward by always rotating left/right
REWARDS = {
        'success' : 10,
        'failure' : -10,
        'step_penalty' : 0,
        'correct_action' : 1
        }

def index_all_items(env):
    """Iterates through each scene and puts all item types into a dict"""
    obj_type_to_index = {}
    for scene_num in AVAILABLE_SCENE_NUMBERS:
        event = env.reset(scene_num)
        for obj in event.metadata['objects']:
            if obj['objectType'] not in obj_type_to_index:
                obj_type_to_index[obj['objectType']] = len(obj_type_to_index)

    return obj_type_to_index

def crow_distance(a, b):
    """
    Returns "as the crow flies" distance between two points a and b,
    specified by dicts with keys 'x', 'y', and 'z'.

    Order of a and b does not matter.
    """
    delta_x = b['x'] - a['x']
    delta_y = b['y'] - a['y']
    delta_z = b['z'] - a['z']

    # Does not account for obstructions, rotation, or looking up/down
    return np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)


class FindOne(object):
    """Task is to find an instance of a specified object in a scene"""
    def __init__(self, env, obj_type_to_index, max_steps=100,
            crow_threshold=None, action_threshold=None,
            walking_threshold=None):
        self.env = env # ThorEnv
        self.obj_type_to_index = obj_type_to_index
        self.max_steps = max_steps
        self.rewards = REWARDS
        self.crow_threshold = crow_threshold
        self.action_threshold = action_threshold
        self.walking_threshold = walking_threshold

    def load_from_traj_data(self, traj_data, high_idx=None):
        if high_idx is None:
            high_idx = 0

        self.traj_data = traj_data
        self.high_idx = high_idx
        self.scene_name_or_num = self.traj_data['scene']['scene_num']

        # From models/eval/eval.py
        object_poses = self.traj_data['scene']['object_poses']
        dirty_and_empty = self.traj_data['scene']['dirty_and_empty']
        object_toggles = self.traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % self.scene_name_or_num
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        self.env.step(dict(self.traj_data['scene']['init_action']))

        # From models/eval/eval_subgoals.py
        expert_init_actions = [a['discrete_action'] for a in
                self.traj_data['plan']['low_actions'] if a['high_idx'] <
                self.high_idx]

        frames = [self.env.last_event.frame]
        for i in range(len(expert_init_actions)):
            action = expert_init_actions[i]
             # get expert action
            compressed_mask = action['args']['mask'] if 'mask' in \
                    action['args'] else None
            mask = self.env.decompress_mask(compressed_mask) if \
                    compressed_mask is not None else None

            # execute expert action
            success, _, _, err, _ = self.env.va_interact(action['action'],
                    interact_mask=mask)
            if not success:
                print("expert initialization failed")
                print(str(err))
                return None, -1

            frames.append(self.env.last_event.frame)

        self.target_object_type = constants.OBJECTS_LOWER_TO_UPPER[
                self.traj_data['plan']['high_pddl'][self.high_idx]
                ['discrete_action']['args'][-1]
        ]
        if self.target_object_type not in self.obj_type_to_index:
            print('Target object type (' + self.target_object_type +
                    ') of trajectory not in obj_type_to_index')
        instance_ids = [obj['objectId'] for obj in
                self.env.last_event.metadata['objects']]
        interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        interactable_objects = [self.env.last_event.get_object(instance_id)
                for instance_id in interactable_instance_ids]
        self.target_objects = [obj for obj in interactable_objects if
                obj['objectType'] == self.target_object_type]
        self.target_instance_ids = [obj['objectId'] for obj in
                self.target_objects]
        self.target_object_type_index = self.obj_type_to_index[
                self.target_object_type]

        # Build scene graph
        self.graph = Graph(use_gt=True, construct_graph=True,
                scene_id=self.scene_name_or_num)
        # Find out which graph points are closest to the object and the ending
        # positions+rotations
        self.end_poses, self.end_point_indexes = \
                self.get_end_poses_point_indexes()

        self.update_expert_actions_path()
        # Expert actions need not be the same as those in traj_data but they
        # almost certainly will be
        self.original_expert_actions = self.current_expert_actions
        self.steps_taken = 0
        self.done = False
        self.success = False

        return frames, self.target_object_type_index

    def reset(self, scene_name_or_num=None):
        if scene_name_or_num is None:
            # Randomly choose a scene if none specified
            scene_name_or_num = random.choice(AVAILABLE_SCENE_NUMBERS)
        self.traj_data = None
        self.high_idx = None
        self.scene_name_or_num = scene_name_or_num
        event = self.env.reset(scene_name_or_num) # Returns ai2thor.server.Event

        # First, pick an object type that's present in the scene and
        # interactable
        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        interactable_objects = [event.get_object(instance_id) for instance_id
                in interactable_instance_ids]
        # Use a set to remove duplicates and avoid bias by number of objects in
        # scene
        interactable_object_types = list(set([obj['objectType'] for obj in
            interactable_objects]))
        self.target_object_type = random.choice(interactable_object_types)
        # Next, make sure it's present in obj_type_to_index
        while self.target_object_type not in self.obj_type_to_index:
            self.target_object_type = random.choice(interactable_object_types)
        # Finally, keep track of all objects in the scene of that type
        self.target_objects = [obj for obj in interactable_objects if
                obj['objectType'] == self.target_object_type]
        self.target_instance_ids = [obj['objectId'] for obj in
                self.target_objects]

        self.target_object_type_index = self.obj_type_to_index[
                self.target_object_type]

        # Build scene graph
        self.graph = Graph(use_gt=True, construct_graph=True,
                scene_id=scene_name_or_num)

        agent_height = event.metadata['agent']['position']['y']
        # Find out which graph points are closest to the object and the ending
        # positions+rotations
        self.end_poses, self.end_point_indexes = \
                self.get_end_poses_point_indexes()

        # Randomly initialize agent position
        # len(self.graph.points) - 1 because randint is inclusive
        start_point_index = random.randint(0, len(self.graph.points) - 1)
        # NOTE: does not account for crow_success, action_success, or
        # walking_success, (but an agent still has to predict "Done" to get
        # credit)
        while start_point_index in self.end_point_indexes:
            start_point_index = random.randint(0, len(self.graph.points) - 1)
        start_point = self.graph.points[start_point_index]
        start_pose = (start_point[0], start_point[1], random.randint(0, 3), 0)
        action = {'action': 'TeleportFull',
                  'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                  'y': agent_height,
                  'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'rotation': start_pose[2],
                  'horizon': start_pose[3],
                  }
        event = self.env.step(action)

        # Get "perfect" trajectory
        self.update_expert_actions_path()
        self.original_expert_actions = self.current_expert_actions
        self.steps_taken = 0
        self.done = False
        self.success = False

        return (event.frame, self.target_object_type_index)

    def step(self, action):
        """
        Figure out "correct" action via Graph, advance agent based on input
        action, then return correct action along with obs, reward, done, info.
        """
        if self.done:
            print('Trying to step() in a done environment!')
            assert False # TODO: debugging assert
            # Return None instead of last event and best action, and action
            # unsuccessful, 0 for reward
            return (None, self.target_object_type), 0, self.done, \
                    (False, None, None)
        # TODO: remove this debugging code
        if len(self.current_expert_actions) == 0:
            print('current_expert_actions has no actions in it!')
            print('scene_name_or_num: ' + str(self.scene_name_or_num))
            print('pose_discrete: ' + str(self.env.last_event.pose_discrete))
            print('end_poses: ' + str(self.end_poses))
            print('target_objects: ' + str(self.target_objects))
            print('target_instance_ids: ' + str(self.target_instance_ids))
            print('steps_taken: ' + str(self.steps_taken))
            if self.traj_data is not None:
                print('task type: ' + self.traj_data['task_type'])
                print('task id: ' + self.traj_data['task_id'])
                print('high idx: ' + str(self.high_idx))
            assert False

        best_action = self.current_expert_actions[0]['action']

        reward = self.rewards['step_penalty']
        if action == best_action:
            reward = self.rewards['correct_action']

        if action == ACTIONS_DONE or self.steps_taken == self.max_steps:
            self.done = True

            if action == ACTIONS_DONE and (((self.crow_success() or
                self.action_success() or self.walking_success()) and
                self.target_visible()) or self.env.last_event.pose_discrete in
                        self.end_poses):
                reward = self.rewards['success']
                self.success = True
            else:
                reward = self.rewards['failure']
                self.success = False

            self.update_expert_actions_path()
            # Currently, return None instead of the last event and action
            # successful
            return (None, self.target_object_type), reward, self.done, \
                    (True, None, best_action)

        # Returns success, event, target_object_type ('' if none),
        # event.metadata['errorMessage'] ('' if none), api_action (action dict
        # with forceAction and action)
        success, event, _, _, _ = self.env.va_interact(action)

        self.update_expert_actions_path()

        self.steps_taken += 1

        return (event.frame, self.target_object_type), reward, self.done, \
            (success, event, best_action) #obs, rew, done, info

    def get_end_poses_point_indexes(self):
        """
        Get the end poses and the indexes of the ending point in the navigation
        graph (self.graph) corresponding to self.target_instance_ids
        """
        agent_height = self.env.last_event.metadata['agent']['position']['y']
        end_poses = []
        end_point_indexes = []
        # TODO: should this to match up perfectly with get_obj_coords in
        # gen/game_states/task_game_state_full_knowledge.py? get_obj_coords
        # uses backed off point if the distance to object is < 0.5 and the
        # point is better than before. This works fine as-is for most objects
        # but more poorly for large things like CounterTop
        for target_object in self.target_objects:
            distances_to_target = []
            for point in self.graph.points:
                point_xyz = {
                        'x' : point[0]*constants.AGENT_STEP_SIZE,
                        'y' : agent_height,
                        'z' : point[1]*constants.AGENT_STEP_SIZE
                }
                distances_to_target.append(crow_distance(
                    target_object['position'], point_xyz))

            end_point_index = np.argmin(distances_to_target)

            end_point = self.graph.points[end_point_index]
            delta_x = target_object['position']['x'] - \
                    end_point[0]*constants.AGENT_STEP_SIZE
            delta_z = target_object['position']['z'] - \
                    end_point[1]*constants.AGENT_STEP_SIZE
            '''
            Center is point, where agent would stand.
                    z
              \     |     /
               \  0 | 0  /
                \   |   /
                 \  |  /
               3  \ | /  1
                   \|/
            ---------------- x
                   /|\
               3  / | \  1
                 /  |  \
                /   |   \
               /  2 | 2  \
              /     |     \

            '''
            if np.abs(delta_x) < np.abs(delta_z) and delta_z >= 0:
                end_rotation = 0
            elif np.abs(delta_x) < np.abs(delta_z) and delta_z < 0:
                end_rotation = 2
            elif np.abs(delta_x) >= np.abs(delta_z) and delta_x >= 0:
                end_rotation = 1
            elif np.abs(delta_x) >= np.abs(delta_z) and delta_x < 0:
                end_rotation = 3
            # Calculate look angle: from get_obj_coords in
            # gen/game_states/task_game_state_full_knowledge.py
            horizontal_dist_to_obj = np.max(np.abs([delta_x, delta_z]))
            # Use the center y for now. Corners are in
            # target_object['objectBounds']['objectBoundsCorners']
            obj_height = agent_height - target_object['position']['y']
            camera_angle = int(np.clip(np.round(np.arctan2(obj_height,
                horizontal_dist_to_obj) * (180 / np.pi /
                    constants.HORIZON_GRANULARITY)) *
                constants.HORIZON_GRANULARITY, -30,  60))

            # Hard overwrites for camera_angle adjustments based on object type
            # and scene.
            if self.target_object_type is not None and self.scene_name_or_num \
                    is not None:
                if (self.target_object_type, self.scene_name_or_num) in \
                        constants.FORCED_HORIZON_OBJS:
                    camera_angle = constants.FORCED_HORIZON_OBJS[
                        (self.target_object_type, self.scene_name_or_num)]
                elif (self.target_object_type, None) in \
                        constants.FORCED_HORIZON_OBJS:
                    camera_angle = constants.FORCED_HORIZON_OBJS[
                        (self.target_object_type, None)]

            end_pose = (end_point[0], end_point[1], end_rotation, camera_angle)
            end_poses.append(end_pose)
            end_point_indexes.append(end_point_index)

        return end_poses, end_point_indexes

    def update_expert_actions_path(self):
        """
        Updates self.current_expert_actions and self.current_expert_path based
        on current agent position and closest target object (by crow's
        distance).
        """
        target_object_distances_to_agent = [crow_distance(
            target_object['position'],
            self.env.last_event.metadata['agent']['position'])
            for target_object in self.target_objects]

        closest_target_object_end_pose = self.end_poses[np.argmin(
            target_object_distances_to_agent)]

        actions, path = self.graph.get_shortest_path(
                self.env.last_event.pose_discrete,
                closest_target_object_end_pose)
        # If there are no expert actions left and the environment is done, then
        # ACTIONS_DONE was taken and should not be appended
        if not (len(actions) == 0 and self.done):
            actions.append({'action' : ACTIONS_DONE})
        self.current_expert_actions = actions
        self.current_expert_path = path

    # Some convenience functions to avoid poking around in FindOne internals
    def get_current_expert_actions_path(self):
        """
        This can be used to get the number of actions between the agent and
        success.
        """
        return self.current_expert_actions, self.current_expert_path

    def get_original_expert_actions(self):
        return self.original_expert_actions

    def get_scene_name_or_num(self):
        return self.scene_name_or_num

    def get_target_object_type(self):
        return self.target_object_type

    def get_success(self):
        return self.success

    def crow_success(self):
        return self.crow_threshold is not None and \
                    self.crow_distance_to_goal() < self.crow_threshold

    def action_success(self):
        return self.action_threshold is not None and \
                    self.action_distance_to_goal() < self.action_threshold

    def walking_success(self):
        return self.walking_threshold is not None and \
                    self.walking_distance_to_goal() < self.walking_threshold

    def crow_distance_to_goal(self):
        target_object_distances_to_agent = [crow_distance(
            target_object['position'],
            self.env.last_event.metadata['agent']['position'])
            for target_object in self.target_objects]

        return min(target_object_distances_to_agent)

    def action_distance_to_goal(self):
        return len(self.current_expert_actions)

    def walking_distance_to_goal(self):
        actions, path = self.get_current_expert_actions_path()
        return len([a for a in actions if a['action'] ==
            'MoveAhead'])

    def target_visible(self):
        """
        Returns 1 if a target is visible from the current pose, 0 otherwise.
        """
        for object_id in self.target_instance_ids:
            if self.env.last_event.get_object(object_id)['visible']:
                return True
        return False

if __name__ == '__main__':
    env = ThorEnv()

    if not os.path.isfile('obj_type_to_index.json'):
        obj_type_to_index = index_all_items(env)
        with open('obj_type_to_index.json', 'w') as jsonfile:
            json.dump(obj_type_to_index, jsonfile)
    else:
        with open('obj_type_to_index.json', 'r') as jsonfile:
            obj_type_to_index = json.load(jsonfile)
    print(obj_type_to_index)
    index_to_obj_type = {i: ot for ot, i in obj_type_to_index.items()}

    fo = FindOne(env, obj_type_to_index)
