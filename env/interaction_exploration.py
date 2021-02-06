import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import random

import cv2
from env.thor_env import ThorEnv
import gen.constants as constants
from gen.graph.graph_obj import Graph

INTERACT_MODE_SIMPLE = 0
INTERACT_MODE_COMPLEX = 1

ACTIONS_INTERACT = 'Interact'
NAV_ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']
INT_ACTIONS = ['OpenObject', 'CloseObject', 'PickupObject', 'PutObject',
        'CleanObject', 'HeatObject', 'CoolObject', 'ToggleObjectOn',
        'ToggleObjectOff', 'SliceObject']
SIMPLE_ACTIONS = NAV_ACTIONS + [ACTIONS_INTERACT]
COMPLEX_ACTIONS = NAV_ACTIONS + INT_ACTIONS
INDEX_TO_ACTION_SIMPLE = dict(enumerate(SIMPLE_ACTIONS))
ACTION_TO_INDEX_SIMPLE = dict((v,k) for k,v in INDEX_TO_ACTION_SIMPLE.items())
INDEX_TO_ACTION_COMPLEX = dict(enumerate(COMPLEX_ACTIONS))
ACTION_TO_INDEX_COMPLEX = dict((v,k) for k,v in
        INDEX_TO_ACTION_COMPLEX.items())

# TODO: fix up reward function and add options for reward for every new state
# or reward only for new interactions
REWARDS = {
        'success' : 10,
        'failure' : -10,
        'step_penalty' : 0,
        'correct_action' : 1,
        'interact' : 1
        }

class InteractionExploration(object):
    """Task is to interact with all objects in a scene"""
    def __init__(self, env, interact_mode=INTERACT_MODE_SIMPLE,
            use_masks=False, max_steps=100):
        self.env = env # ThorEnv
        self.interact_mode = interact_mode
        self.use_masks = use_masks
        self.max_steps = max_steps
        self.rewards = REWARDS
        self.done = False

    def reset(self, scene_name_or_num=None):
        if scene_name_or_num is None:
            # Randomly choose a scene if none specified
            scene_name_or_num = random.choice(constants.SCENE_NUMBERS)
        event = self.env.reset(scene_name_or_num)

        # Tabulate all interactable objects and mark as uninteracted with
        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        self.objects_interacted = {instance_id : False for instance_id in
                interactable_instance_ids}

        # Build scene graph
        self.graph = Graph(use_gt=True, construct_graph=True,
                scene_id=scene_name_or_num)

        # Randomly initialize agent position
        # len(self.graph.points) - 1 because randint is inclusive
        start_point_index = random.randint(0, len(self.graph.points) - 1)
        start_point = self.graph.points[start_point_index]
        start_pose = (start_point[0], start_point[1], random.randint(0, 3), 0)
        action = {'action': 'TeleportFull',
                  'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                  'y': event.metadata['agent']['position']['y'],
                  'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'rotation': start_pose[2],
                  'horizon': start_pose[3],
                  }
        event = self.env.step(action)

        # TODO: make a function that gets the closest object and computes the
        # path to the closest object, copy over expert actions, and add
        # Interact action (optionally with mask) or other appropriate action to
        # end of expert_actions
        self.steps_taken = 0
        self.done = False

        return event.frame

    def step(self, action, interact_mask=None):

        reward = self.rewards['step_penalty']
        # Reject action if already done
        if self.done:
            return None, reward, self.done, (False, self.env.last_event)

        # TODO: if Interact simple action taken, get closest visible object
        # that is interactable and apply correct action

        if action == ACTIONS_INTERACT and self.interact_mode == \
                INTERACT_MODE_SIMPLE:
            #self.env.to_thor_api_exec(specific_action, object_id=object_id,
            #    smooth_nav=True)
            pass


        # Returns success, event, target_instance_id ('' if none),
        # event.metadata['errorMessage'] ('' if none), api_action (action dict
        # with forceAction and action)
        success, event, target_instance_id, _, _ = self.env.va_interact(action,
                interact_mask=interact_mask)

        if target_instance_id != '':
            if not self.objects_interacted[target_instance_id]:
                reward = self.rewards['interact']
                self.objects_interacted[target_instance_id] = True

            if all(self.objects_interacted.values()):
                self.done = True

        self.steps_taken += 1

        # obs, rew, done, info
        return event.frame, reward, self.done, (success, event)

    def closest_object(self, uninteracted=False):
        """
        Returns object id of closest interactable object to current agent
        position. If uninteracted is True, will only return closest
        uninteracted object.
        """
        agent_x, agent_y, agent_z, _ = self.env.last_event.pose_discrete
        closest_object_id = None
        closest_object_distance = float('inf')
        for object_id, object_interacted in self.objects_interacted:
            if uninteracted and object_interacted:
                continue
            distance = self.env.last_event.get_object(object_id)['distance']
            if closest_object_id is None or distance < closest_object_distance:
                closest_object_id = object_id
                closest_object_distance = distance
        return closest_object_id

if __name__ == '__main__':
    env = ThorEnv()

    ie = InteractionExploration(env)
    ie.reset()
