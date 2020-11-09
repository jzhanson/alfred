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
ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'Done']
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
        event = self.env.reset(scene_name_or_num) # Returns ai2thor.server.Event

        # Pick an interactable object in the scene to go find
        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        self.target_instance_id = random.choice(interactable_instance_ids)
        for obj in event.metadata['objects']:
            if obj['objectId'] == self.target_instance_id:
                self.target_instance_index = self.obj_type_to_index[obj[
                    'objectType']]
                target_object = obj

        # Build scene graph
        self.graph = Graph(use_gt=True, construct_graph=True,
                scene_id=scene_name_or_num)

        agent_height = event.metadata['agent']['position']['y']
        # Find out which graph point is closest to the object is the closest
        # and which way the agent should face
        distances_to_target = []
        for point in self.graph.points:
            delta_x = target_object['position']['x'] - \
                    point[0]*constants.AGENT_STEP_SIZE
            delta_y = target_object['position']['y'] - agent_height
            delta_z = target_object['position']['z'] - \
                    point[1]*constants.AGENT_STEP_SIZE
            distances_to_target.append(np.sqrt(delta_x**2 + delta_y**2 +
                delta_z**2))

        end_point_index = np.argmin(distances_to_target)

        end_point = self.graph.points[end_point_index]
        delta_x = end_point[0]*constants.AGENT_STEP_SIZE - \
                target_object['position']['x']
        delta_z = end_point[1]*constants.AGENT_STEP_SIZE - \
                target_object['position']['z']
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
        self.end_pose = (end_point[0], end_point[1], end_rotation, 0)

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
        expert_actions, expert_path = self.graph.get_shortest_path(
                start_pose, self.end_pose)
        # Add Done action to end of expert_actions
        expert_actions.append({'action' : ACTIONS_DONE})
        self.original_expert_actions = expert_actions
        self.steps_taken = 0

        self.current_expert_actions = expert_actions
        self.current_expert_path = expert_path

        return (event.frame, self.target_instance_index)

    def step(self, action):
        """
        Figure out "correct" action via Graph, advance agent based on input
        action, then return correct action along with obs, reward, done, info.
        """
        best_action = self.current_expert_actions[0]['action']

        reward = self.rewards['step_penalty']
        # TODO: reject action if already done
        done = False
        if action == best_action:
            reward = self.rewards['correct_action']

        if action == ACTIONS_DONE or self.steps_taken == self.max_steps:
            done = True
            if action == ACTIONS_DONE and best_action == ACTIONS_DONE:
                reward = self.rewards['success']
            else:
                reward = self.rewards['failure']
            # Currently, return None instead of the last event
            return (None, self.target_instance_index), reward, done, (True,
                    None, best_action)

        # Returns success, event, target_instance_id ('' if none),
        # event.metadata['errorMessage'] ('' if none), api_action (action dict
        # with forceAction and action)
        success, event, _, _, _ = self.env.va_interact(action)

        actions, path = self.graph.get_shortest_path(
                self.env.last_event.pose_discrete, self.end_pose)
        actions.append({'action' : ACTIONS_DONE})
        self.current_expert_actions = actions
        self.current_expert_path = path

        self.steps_taken += 1

        return (event.frame, self.target_instance_index), reward, done, \
            (success, event, best_action) #obs, rew, done, info

    # Some convenience functions to avoid poking around in FindOne internals
    def get_current_expert_actions_path(self):
        """
        This can be used to get the number of actions between the agent and
        success.
        """
        return self.current_expert_actions, self.current_expert_path

    def crow_distance_to_goal(self):
        agent_x, agent_y, _, _ = self.env.last_event.pose_discrete
        end_x, end_y, _, _ = self.end_pose
        # Does not account for obstructions, rotation, or looking up/down
        return abs(agent_x - end_x) + abs(agent_y - end_y)

    def walking_distance_to_goal(self):
        actions, path = self.get_current_expert_actions_path()
        return len([a for a in actions if a['action'] ==
            'MoveAhead'])

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
