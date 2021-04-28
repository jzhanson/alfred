import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import random

import numpy as np

import cv2
from env.thor_env import ThorEnv
import gen.constants as constants
from gen.graph.graph_obj import Graph
from env.reward import InteractionReward

class InteractionExploration(object):
    """Task is to interact with all objects in a scene"""
    def __init__(self, env, reward, single_interact=False,
            sample_contextual_action=False, use_masks=False):
        """Initialize environment

        single_interact enables the single "Interact" action for contextual
        interaction.
        """
        self.env = env # ThorEnv
        self.single_interact = single_interact
        self.sample_contextual_action = sample_contextual_action
        self.use_masks = use_masks
        self.reward = reward
        self.done = False
        self.scene_name_or_num = None

    def reset(self, scene_name_or_num=None, random_object_positions=True,
            random_position=True, random_rotation=True,
            random_look_angle=True):
        if scene_name_or_num is None:
            # Randomly choose a scene if none specified
            scene_name_or_num = random.choice(constants.SCENE_NUMBERS)

        self.scene_name_or_num = scene_name_or_num
        event = self.env.reset(self.scene_name_or_num)

        if random_object_positions:
            # Can play around with numDuplicatesOfType to populate the
            # environment with more objects
            #
            # TODO: can also consider randomizing object states (e.g.
            # open/closed, toggled)
            event = self.env.step(dict(action='InitialRandomSpawn',
                    randomSeed=0,
                    forceVisible=False,
                    numPlacementAttempts=5,
                    placeStationary=True,
                    numDuplicatesOfType=None, #[{objType: count}]
                    excludedReceptacles=None))

        # Tabulate all interactable objects and mark as uninteracted with
        instance_ids = [obj['objectId'] for obj in event.metadata['objects']]
        self.interactable_instance_ids = self.env.prune_by_any_interaction(
                instance_ids)
        self.objects_interacted = {instance_id : False for instance_id in
                self.interactable_instance_ids}

        # Build scene graph
        self.graph = Graph(use_gt=True, construct_graph=True,
                scene_id=self.scene_name_or_num)

        start_point = event.pose_discrete[:2]
        rotation = event.pose_discrete[2]
        look_angle = event.pose_discrete[3]
        if random_position:
            # Randomly initialize agent position
            # len(self.graph.points) - 1 because randint is inclusive
            start_point_index = random.randint(0, len(self.graph.points) - 1)
            start_point = self.graph.points[start_point_index]
        if random_rotation:
            rotation = random.randint(0, 3)
        if random_look_angle:
            look_angle = random.randrange(-30, 61, 15) # Include 60 degrees
        start_pose = (start_point[0], start_point[1], rotation, look_angle)
        action = {'action': 'TeleportFull',
                  'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                  'y': event.metadata['agent']['position']['y'],
                  'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'rotation': start_pose[2] * 90,
                  'horizon': start_pose[3],
                  }
        event = self.env.step(action)

        self.reward.reset(scene_name_or_num=self.scene_name_or_num)
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
        """Advances environment based on given action and mask.
        """
        # Reject action if already done
        if self.done:
            err = 'Trying to step in a done environment'
            success = False
            # By default this will return invalid navigation reward, but the
            # reward doesn't matter here, it's an edge case
            return (self.env.last_event.frame, self.reward.invalid_action(),
                    self.done, (success, self.env.last_event, err))

        is_interact_action = (action == constants.ACTIONS_INTERACT or action in
                constants.INT_ACTIONS)

        # If using masks, a mask must be provided with an interact action
        if self.use_masks and is_interact_action and interact_mask is None:
            err = 'No mask provided with interact action ' + action
            success = False
            return (self.env.last_event.frame,
                    self.reward.invalid_action(interaction=True), self.done,
                    (success, self.env.last_event, err))

        # If not using masks, have to choose an object based on camera
        # view and proximity and interactability
        if is_interact_action and not self.use_masks:
            # Choose object
            # TODO: can try out projecting the point at the center of the
            # screen, and finding the object closest to that
            target_instance_id = self.center_of_view_object(
                    allow_interacted=True, contextual=True)
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
                        return (self.env.last_event.frame,
                                self.reward.invalid_action(interaction=True),
                                self.done, (success, self.env.last_event, err))
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
                        return (self.env.last_event.frame,
                                self.reward.invalid_action(interaction=True),
                                self.done, (success, self.env.last_event, err))
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
                self.objects_interacted[target_instance_id] = True

            if all(self.objects_interacted.values()):
                self.done = True

        '''
        print('target instance id: ', target_instance_id)
        print([obj['objectId'] for obj in
            self.env.last_event.metadata['objects'] if obj['visible']])
        '''

        self.steps_taken += 1
        reward = self.reward.get_reward(self.env.last_event, action,
                api_success=success, target_instance_id=target_instance_id,
                interact_mask=interact_mask)

        # obs, rew, done, info
        return self.env.last_event.frame, reward, self.done, (success,
                self.env.last_event, err)

    def contextual_attributes(self):
        """Get attributes of interactable objects based on agent state.

        """
        # TODO: Unused properties: dirtyable, breakable, cookable,
        # canFillWithLiquid, canChangeTempToCold, canChangeTempToHot,
        # canBeUsedUp
        # env/thor_env.py takes care of cleaned, heated, and cooled objects by
        # keeping a list
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
        return contextual_attributes

    def center_of_view_object(self, allow_interacted=True, contextual=True):
        """Get object at or closest to center of view.
        """
        if contextual:
            contextual_attributes = self.contextual_attributes()
        inventory_object_id = (self.env.last_event.metadata
                ['inventoryObjects'][0]['objectId'] if
                len(self.env.last_event.metadata['inventoryObjects']) > 0 else
                None)

        center_x = self.env.last_event.screen_width / 2
        center_y = self.env.last_event.screen_height / 2

        visible_object_ids = [obj['objectId'] for obj in
                self.env.last_event.metadata['objects'] if obj['visible']]

        object_id_to_average_pixel_distance = {}

        center_of_view_object_id = None
        for object_id, object_interacted in self.objects_interacted.items():
            obj = self.env.last_event.get_object(object_id)
            if not obj['visible']:
                continue
            if not allow_interacted and object_interacted:
                continue
            if contextual and not any([obj[attribute] for attribute in
                contextual_attributes]):
                continue
            if inventory_object_id == object_id:
                continue
            mask = self.env.last_event.instance_masks[object_id]
            xs, ys = np.nonzero(mask)
            xs_distance = np.power(xs - center_x, 2)
            ys_distance = np.power(ys - center_y, 2)
            object_id_to_average_pixel_distance[object_id] = np.mean(np.sqrt(
                xs_distance + ys_distance))
        sorted_object_id_pixel_distances = sorted(
                object_id_to_average_pixel_distance.items(), key=lambda x:
                x[1])
        if len(sorted_object_id_pixel_distances) == 0:
            return None
        center_of_view_object_id = sorted_object_id_pixel_distances[0][0]

        return center_of_view_object_id

    def closest_object(self, allow_not_visible=False,
        allow_interacted=True, contextual=True):
        """
        Returns object id of closest visible interactable object to current
        agent position.

        If contextual is true, items will be filtered based on current state
        (e.g. if not holding anything, will allow pickupable items, if holding
        a knife, will allow sliceable items).

        If allow_interacted is False, will only return closest visible
        uninteracted object.

        Inventory objects are not counted.
        """
        # If one of the attributes is true, then the object is included
        if contextual:
            contextual_attributes = self.contextual_attributes
        inventory_object_id = (self.env.last_event.metadata
                ['inventoryObjects'][0]['objectId'] if
                len(self.env.last_event.metadata['inventoryObjects']) > 0 else
                None)

        # Return None (not '') if no object is found because this function will
        # be called when trying to get an object for contextual interaction,
        # meaning no object is found rather than no target needed for ''
        closest_object_id = None
        closest_object_distance = float('inf')
        for object_id, object_interacted in self.objects_interacted.items():
            obj = self.env.last_event.get_object(object_id)
            if not allow_not_visible and not obj['visible']:
                continue
            if not allow_interacted and object_interacted:
                continue
            if inventory_object_id == object_id:
                continue
            if contextual and not any([obj[attribute] for attribute in
                contextual_attributes]):
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

        Due to limitations with a conditional statement, can only do one action
        if multiple apply, such as opening or toggling a microwave. If sampling
        from all valid actions is desired, set
        self.sample_contextual_action=True.
        """
        obj = self.env.last_event.get_object(target_instance_id)
        holding_object = len(self.env.last_event.metadata['inventoryObjects']) \
                > 0
        held_object = self.env.last_event.metadata['inventoryObjects'][0] if \
                holding_object else None
        valid_actions = []
        if obj['openable'] and not obj['isOpen']:
            valid_actions.append('OpenObject')
        # Favor putting object over repeatedly opening/closing object
        if obj['receptacle'] and holding_object:
            valid_actions.append('PutObject')
        if obj['openable'] and obj['isOpen']:
            valid_actions.append('CloseObject')
        if obj['toggleable'] and not obj['isToggled']:
            valid_actions.append('ToggleObjectOn')
        if obj['toggleable'] and obj['isToggled']:
            valid_actions.append('ToggleObjectOff')
        if obj['pickupable'] and not holding_object:
            valid_actions.append('PickupObject')
        if holding_object and 'Knife' in held_object['objectType'] and \
                obj['sliceable']:
            valid_actions.append('SliceObject')

        if len(valid_actions) > 0:
            return (random.choice(valid_actions) if
                    self.sample_contextual_action else valid_actions[0])
        else:
            # Sometimes there won't be a valid interaction
            return None

    def get_scene_name_or_num(self):
        return self.scene_name_or_num

    def get_current_expert_actions_path(self):
        # TODO: make a function that gets the closest object and computes the
        # path to the closest object, copy over expert actions, and add
        # Interact action (optionally with mask) or other appropriate action to
        # end of expert_actions
        return [{'action' : 'MoveAhead'}], []

    def get_coverages(self):
        return self.reward.get_coverages()

if __name__ == '__main__':
    single_interact = False
    use_masks = True
    accept_input = True
    env = ThorEnv()

    # For showing sample superpixels
    from skimage.segmentation import slic, mark_boundaries
    from skimage.util import img_as_float
    slic_kwargs = {
            'max_iter' : 10,
            'spacing' : None,
            'multichannel' : True,
            'convert2lab' : True,
            'enforce_connectivity' : True,
            'max_size_factor' : 3,
            'n_segments' : 10,
            'compactness' : 10.0,
            'sigma' : 0,
            'min_size_factor' : 0.01
    }

    import json
    with open(os.path.join(os.environ['ALFRED_ROOT'], 'models',
        'config', 'rewards.json'), 'r') as jsonfile:
        reward_config = json.load(jsonfile)['InteractionExploration']

    reward = InteractionReward(env, reward_config, repeat_discount=0.99,
            persist_state=True, use_novelty=True)

    ie = InteractionExploration(env, reward, single_interact=single_interact,
            use_masks=use_masks)
    frame = ie.reset(random_object_positions=False, random_position=False,
            random_rotation=False, random_look_angle=False)
    done = False
    import numpy as np
    while not done:
        cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_frame.png'), cv2.cvtColor(ie.env.last_event.frame,
                cv2.COLOR_BGR2RGB))
        segments = slic(img_as_float(frame), **slic_kwargs)
        cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_superpixels.png'),
            mark_boundaries(cv2.cvtColor(ie.env.last_event.frame,
                cv2.COLOR_BGR2RGB), segments) * 255)
        cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_segs.png'), cv2.cvtColor(
                ie.env.last_event.instance_segmentation_frame,
                cv2.COLOR_BGR2RGB))

        if accept_input:
            key_action = input("Input action: ")
            action = (constants.KEY_TO_ACTION[key_action] if key_action in
                    constants.KEY_TO_ACTION else key_action)
            if action == 'reset':
                scene_num = input("Input new scene num, nothing for random: ")
                if len(scene_num) == 0:
                    scene_num = None
                else:
                    scene_num = int(scene_num)
                frame = ie.reset(scene_name_or_num=scene_num,
                        random_object_positions=False, random_position=False,
                        random_rotation=False, random_look_angle=False)
                continue
            if action == 'pose':
                pose_0, pose_1, pose_2, pose_3 = (input("Input 4-tuple pose: ")
                        .split(' '))
                action = {'action': 'TeleportFull',
                        'x': int(pose_0) * constants.AGENT_STEP_SIZE,
                        'y': env.last_event.metadata['agent']['position']['y'],
                        'z': int(pose_1) * constants.AGENT_STEP_SIZE,
                        'rotateOnTeleport': True,
                        'rotation': int(pose_2) * 90,
                        'horizon': int(pose_3),
                        }
                event = env.step(action)
                continue
        else:
            action = random.choice(constants.SIMPLE_ACTIONS if single_interact
                    else constants.COMPLEX_ACTIONS)

        if (action == constants.ACTIONS_INTERACT or action in
                constants.INT_ACTIONS) and use_masks:
            if accept_input:
                mask_x, mask_y = (input("Input x and y of pixel for 'mask': ")
                        .split(' '))
                chosen_object_mask = np.zeros((300, 300))
                # Can also specify 4 numbers for a bounding box, or select the
                # ground truth segmentation mask using the provided pixel
                chosen_object_mask[int(mask_y)][int(mask_x)] = 1
            else:
                # Choose a random mask of an interactable object
                # Sometimes the inventory object may be chosen, but that's fine
                # for this testing code
                visible_objects = ie.env.prune_by_any_interaction(
                        [obj['objectId'] for obj in
                            ie.env.last_event.metadata['objects'] if
                            obj['visible']])
                if len(visible_objects) == 0:
                    print('Attempting to interact but no visible objects')
                    chosen_object_mask = None
                else:
                    chosen_object = random.choice(visible_objects)
                    chosen_object_mask = ie.env.last_event \
                            .instance_masks[chosen_object]
        else:
            chosen_object_mask = None
        frame, reward, done, (success, event, err) = ie.step(action,
                interact_mask=chosen_object_mask)
        print('current pose:', event.pose_discrete)

        if chosen_object_mask is not None:
            mask_image = np.zeros((300, 300, 3))
            mask_image[:, :, :] = chosen_object_mask[:, :, np.newaxis] == 1
            mask_image *= 255
            cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
                'test_mask.png'), mask_image)
        print(env.last_event.metadata['lastAction'])

        print(action, success, reward, err)


