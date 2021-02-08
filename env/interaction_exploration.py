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

ACTIONS_INTERACT = 'Interact'
NAV_ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']
INT_ACTIONS = ['OpenObject', 'CloseObject', 'PickupObject', 'PutObject',
        'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']
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

# TODO: track object states in a vector to issue rewards based on novelty
class InteractionExploration(object):
    """Task is to interact with all objects in a scene"""
    def __init__(self, env, single_interact=False, use_masks=False):
        self.env = env # ThorEnv
        self.single_interact = single_interact
        self.use_masks = use_masks
        self.rewards = REWARDS
        self.done = False

    def reset(self, scene_name_or_num=None):
        if scene_name_or_num is None:
            # Randomly choose a scene if none specified
            scene_name_or_num = random.choice(constants.SCENE_NUMBERS)
        event = self.env.reset(scene_name_or_num)

        # Tabulate all interactable objects and mark as uninteracted with
        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        self.interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        self.objects_interacted = {instance_id : False for instance_id in
                self.interactable_instance_ids}

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

    def exec_targeted_action(self, action, target_instance_id):
        """
        Wrapper function to call self.env.to_thor_api_exec and catch exceptions
        to pass back as error strings, like va_interact in env/thor_env.py.
        """
        try:
            event, _ = self.env.to_thor_api_exec(action, target_instance_id,
                    smooth_nav=True)
        except Exception as e:
            err = str(e)
            success = False
            event = self.env.last_event
        else:
            err = event.metadata['errorMessage']
            success = event.metadata['lastActionSuccess']
        return event, success, err


    def step(self, action, interact_mask=None):
        reward = self.rewards['step_penalty']
        # Reject action if already done
        if self.done:
            err = 'Trying to step in a done environment'
            success = False
            return self.env.last_event.frame, reward, self.done, (success,
                    self.env.last_event, err)

        is_interact_action = (action == ACTIONS_INTERACT or action in
                INT_ACTIONS)

        # If using masks, a mask must be provided with an interact action
        if self.use_masks and is_interact_action and interact_mask is None:
            err = 'No mask provided with interact action ' + action
            success = False
            return self.env.last_event.frame, reward, self.done, (success,
                    self.env.last_event, err)

        # If not using masks, have to choose an object based on camera
        # view and proximity and interactability
        if is_interact_action and not self.use_masks:
            # Choose object
            target_instance_id = self.closest_object(allow_not_visible=False,
                    allow_uninteracted=True, contextual=True)
            if target_instance_id is None:
                err = 'No valid object visible for no mask interaction'
                success = False
            else:
                if self.single_interact:
                    # Figure out which action based on the object
                    contextual_action = self.contextual_action(
                            target_instance_id)
                    if contextual_action is None:
                        err = ('No valid contextual interaction for object ' +
                                target_instance_id)
                        success = False
                        return self.env.last_event.frame, reward, self.done, \
                            (success, self.env.last_event, err)
                else:
                    contextual_action = action
                event, success, err = self.exec_targeted_action(
                        contextual_action, target_instance_id)
        else:
            if is_interact_action and self.single_interact:
                # Choose object based on provided mask, then choose an action
                # for that object based on state
                target_instance_id = self.env.mask_to_target_instance_id(
                        interact_mask)
                if target_instance_id is None:
                    err = ("Bad interact mask. Couldn't locate target object"
                            " to determine contextual Interact")
                    success = False
                else:
                    contextual_action = self.contextual_action(
                            target_instance_id)
                    if contextual_action is None:
                        err = ('No valid contextual interaction for object ' +
                                target_instance_id)
                        success = False
                        return self.env.last_event.frame, reward, self.done, \
                            (success, self.env.last_event, err)
                    # Could call env/thor_env.py's va_interact, for some nice
                    # debug code
                    #success, event, target_instance_id, err, _ = \
                    #        self.env.va_interact(action,
                    #                interact_mask=interact_mask)
                    event, success, err = self.exec_targeted_action(
                            contextual_action, target_instance_id)
            else:
                if not is_interact_action and interact_mask is not None:
                    print('Providing interact mask on a non-interact action ' +
                            action + ', setting mask to None')
                    interact_mask = None
                # Returns success, event, target_instance_id ('' if none),
                # event.metadata['errorMessage'] ('' if none), api_action
                # (action dict with forceAction and action)
                success, event, target_instance_id, err, _ = \
                        self.env.va_interact(action,
                                interact_mask=interact_mask)

        # If target_instance_id is None it means no target instance was found,
        # if target_instance_id is '' it means that the action does not require
        # a target
        if target_instance_id is not None and target_instance_id != '':
            if not self.objects_interacted[target_instance_id]:
                reward = self.rewards['interact']
                self.objects_interacted[target_instance_id] = True

            if all(self.objects_interacted.values()):
                self.done = True

        self.steps_taken += 1

        # obs, rew, done, info
        return self.env.last_event.frame, reward, self.done, (success,
                self.env.last_event, err)

    def closest_object(self, allow_not_visible=False,
        allow_uninteracted=True, contextual=True):
        """
        Returns object id of closest visible interactable object to current
        agent position.

        If contextual is true, items will be filtered based on current state
        (e.g. if not holding anything, will allow pickupable items, if holding
        a knife, will allow sliceable items).

        If uninteracted is True, will only return closest visible
        uninteracted object.

        Inventory objects are not counted.
        """
        # If one of the attributes is true, then the object is included
        # TODO: Unused properties: dirtyable, breakable, cookable,
        # canFillWithLiquid, canChangeTempToCold, canChangeTempToHot,
        # canBeUsedUp
        # env/thor_env.py takes care of cleaned, heated, and cooled objects by
        # keeping a list
        # TODO: maybe update env/thor_env.py to also modify object states?
        contextual_attributes = ['openable', 'toggleable']
        if len(self.env.last_event.metadata['inventoryObjects']) > 0:
            contextual_attributes.append('receptacle')
            if 'Knife' in self.env.last_event.metadata['inventoryObjects'][0][
                    'objectType']:
                contextual_attributes.append('sliceable')
        else:
            # Agent is allowed to pick up an item only if it is not holding an
            # item
            # Cleanable, heatable and coolable objects should all be pickupable
            contextual_attributes.append('pickupable')

        inventory_object_id = self.env.last_event.metadata \
                ['inventoryObjects'][0]['objectId'] if \
                len(self.env.last_event.metadata['inventoryObjects']) > 0 else None

        # Return None (not '') if no object is found because this function will
        # be called when trying to get an object for contextual interaction,
        # meaning no object is found rather than no target needed for ''
        closest_object_id = None
        closest_object_distance = float('inf')
        for object_id, object_interacted in self.objects_interacted.items():
            obj = self.env.last_event.get_object(object_id)
            if not allow_not_visible and not obj['visible']:
                continue
            if not allow_uninteracted and object_interacted:
                continue
            if inventory_object_id == object_id:
                continue
            possesses_contextual_attributes = [obj[attribute] for attribute in
                    contextual_attributes]
            if not any(possesses_contextual_attributes):
                continue

            distance = obj['distance']
            if closest_object_id is None or distance < closest_object_distance:
                closest_object_id = object_id
                closest_object_distance = distance
        return closest_object_id

    def contextual_action(self, target_instance_id):
        """
        Returns action for the object with the given id based on object
        attributes.
        """
        obj = self.env.last_event.get_object(target_instance_id)
        holding_object = len(self.env.last_event.metadata['inventoryObjects']) \
                > 0
        held_object = self.env.last_event.metadata['inventoryObjects'][0] if \
                holding_object else None
        # TODO: test this contextual actions
        if obj['openable'] and not obj['isOpen']:
            return 'OpenObject'
        elif obj['receptacle'] and holding_object:
            # and not (obj['openable'] and not obj['isOpen']):
            return 'PutObject'
        elif obj['openable'] and obj['isOpen']:
            return 'CloseObject'
        elif obj['toggleable'] and not obj['isToggled']:
            return 'ToggleObjectOn'
        elif obj['toggleable'] and obj['isToggled']:
            return 'ToggleObjectOff'
        # TODO: are there any/do we want to be able to pick up openable or toggleable items?
        elif obj['pickupable'] and not holding_object:
            return 'PickupObject'
        elif holding_object and 'Knife' in held_object['objectType'] and \
                obj['sliceable']:
            return 'SliceObject'
        else:
            # Sometimes there won't be a valid interaction
            return None

if __name__ == '__main__':
    env = ThorEnv()

    ie = InteractionExploration(env, single_interact=True, use_masks=True)
    frame = ie.reset()
    done = False
    import matplotlib.pyplot as plt
    import numpy as np
    while not done:
        plt.imshow(frame)
        plt.savefig('/home/jzhanson/alfred/alfred_frame.png')
        action = random.choice(SIMPLE_ACTIONS)
        if action == ACTIONS_INTERACT or action in INT_ACTIONS:
            # Choose a random mask of an interactable object
            visible_objects = ie.env.prune_by_any_interaction(
                    [obj['objectId'] for obj in
                        ie.env.last_event.metadata['objects'] if obj['visible']])
            if len(visible_objects) == 0:
                chosen_object_mask = None
            else:
                chosen_object = random.choice(visible_objects)
                # TODO: choose largest mask?
                object_id_to_color = {v:k for k,v in ie.env.last_event.color_to_object_id.items()}
                chosen_object_color = object_id_to_color[chosen_object]
                # np.equal returns (300, 300, 3) despite broadcasting, but all the
                # last dimension are the same
                chosen_object_mask = np.equal(
                        ie.env.last_event.instance_segmentation_frame,
                        chosen_object_color)[:, :, 0]
        else:
            chosen_object_mask = None
        frame, reward, done, (success, event, err) = ie.step(action,
                interact_mask=chosen_object_mask)

        print(action, success, err)


