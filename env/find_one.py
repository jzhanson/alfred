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


class FindOne(object):
    """Task is to find an instance of a specified object in a scene"""
    def __init__(self, env, obj_type_to_index, max_steps=100):
        self.env = env # ThorEnv
        self.obj_type_to_index = obj_type_to_index
        self.max_steps = max_steps
        self.rewards = REWARDS

    def reset(self, scene_name_or_num=None):
        if scene_name_or_num is None:
            # Randomly choose a scene if none specified
            scene_name_or_num = random.choice(AVAILABLE_SCENE_NUMBERS)
        self.scene_name_or_num = scene_name_or_num
        event = self.env.reset(scene_name_or_num) # Returns ai2thor.server.Event

        # Pick an interactable object in the scene to go find that also resides
        # in obj_type_to_index
        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        self.target_instance_id = random.choice(interactable_instance_ids)
        self.target_object = event.get_object(self.target_instance_id)
        while self.target_object['objectType'] not in self.obj_type_to_index:
            self.target_instance_id = random.choice(interactable_instance_ids)
            self.target_object = event.get_object(self.target_instance_id)

        self.target_instance_index = self.obj_type_to_index[self.target_object[
            'objectType']]

        # Build scene graph
        self.graph = Graph(use_gt=True, construct_graph=True,
                scene_id=scene_name_or_num)

        agent_height = event.metadata['agent']['position']['y']
        # Find out which graph point is closest to the object and which way the
        # agent should face
        self.end_pose, end_point_index = self.get_end_pose_point_index()

        # Randomly initialize agent position
        # len(self.graph.points) - 1 because randint is inclusive
        start_point_index = random.randint(0, len(self.graph.points) - 1)
        while start_point_index == end_point_index:
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

        return (event.frame, self.target_instance_index)

    def step(self, action):
        """
        Figure out "correct" action via Graph, advance agent based on input
        action, then return correct action along with obs, reward, done, info.
        """
        best_action = self.current_expert_actions[0]['action']

        reward = self.rewards['step_penalty']
        if self.done:
            # Return None instead of last event and best action, and action
            # unsuccessful
            return (None, self.target_instance_index), reward, self.done, \
                    (False, None, None)
        if action == best_action:
            reward = self.rewards['correct_action']

        if action == ACTIONS_DONE or self.steps_taken == self.max_steps:
            self.done = True
            if action == ACTIONS_DONE and best_action == ACTIONS_DONE:
                reward = self.rewards['success']
            else:
                reward = self.rewards['failure']

            self.update_expert_actions_path()
            # Currently, return None instead of the last event and action
            # successful
            return (None, self.target_instance_index), reward, self.done, \
                    (True, None, best_action)

        # Returns success, event, target_instance_id ('' if none),
        # event.metadata['errorMessage'] ('' if none), api_action (action dict
        # with forceAction and action)
        success, event, _, _, _ = self.env.va_interact(action)

        self.update_expert_actions_path()

        self.steps_taken += 1

        return (event.frame, self.target_instance_index), reward, self.done, \
            (success, event, best_action) #obs, rew, done, info

    def get_end_pose_point_index(self):
        """
        Get the end pose and the index of the ending point in the navigation
        graph (self.graph) corresponding to self.target_instance_id
        """
        agent_height = self.env.last_event.metadata['agent']['position']['y']
        distances_to_target = []
        for point in self.graph.points:
            delta_x = self.target_object['position']['x'] - \
                    point[0]*constants.AGENT_STEP_SIZE
            delta_y = self.target_object['position']['y'] - agent_height
            delta_z = self.target_object['position']['z'] - \
                    point[1]*constants.AGENT_STEP_SIZE
            distances_to_target.append(np.sqrt(delta_x**2 + delta_y**2 +
                delta_z**2))

        end_point_index = np.argmin(distances_to_target)

        end_point = self.graph.points[end_point_index]
        delta_x = end_point[0]*constants.AGENT_STEP_SIZE - \
                self.target_object['position']['x']
        delta_z = end_point[1]*constants.AGENT_STEP_SIZE - \
                self.target_object['position']['z']
        '''
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
        # TODO: figure up ending look up/down?
        #constants.VISIBILITY_DISTANCE
        end_pose = (end_point[0], end_point[1], end_rotation, 0)

        return end_pose, end_point_index

    def update_expert_actions_path(self):
        """
        Updates self.current_expert_actions and self.current_expert_path based
        on current agent position.
        """
        actions, path = self.graph.get_shortest_path(
                self.env.last_event.pose_discrete, self.end_pose)
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

    def crow_distance_to_goal(self):
        delta_x = self.target_object['position']['x'] - \
            self.env.last_event.metadata['agent']['position']['x']
        delta_y = self.target_object['position']['y'] - \
            self.env.last_event.metadata['agent']['position']['y']
        delta_z = self.target_object['position']['z'] - \
            self.env.last_event.metadata['agent']['position']['z']

        # Does not account for obstructions, rotation, or looking up/down
        return delta_x**2 + delta_y**2 + delta_z**2

    def walking_distance_to_goal(self):
        actions, path = self.get_current_expert_actions_path()
        return len([a for a in actions if a['action'] ==
            'MoveAhead'])

    def target_visible(self):
        """
        Returns 1 if the target is visible from the current pose, 0 otherwise.
        """
        instance_segs = self.env.last_event.instance_segmentation_frame
        color_to_object_id = self.env.last_event.color_to_object_id

        scene_object_ids = [obj['objectId'] for obj in
                self.env.last_event.metadata['objects']]

        visible_object_ids = set()
        for i in range(instance_segs.shape[0]):
            for j in range(instance_segs.shape[1]):
                object_id = color_to_object_id[tuple(instance_segs[i, j])]
                # Some segmentations correspond to object types? For example
                # some are 'Sink|-01.99|+01.14|-00.98|SinkBasin', 'SinkBasin',
                # 'Sink', 'Sink|-01.99|+01.14|-00.98'
                if object_id in scene_object_ids:
                    visible_object_ids.add(object_id)
        return int(self.target_object['objectId'] in visible_object_ids)

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
