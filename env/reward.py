from gen.utils.game_util import get_object
import gen.constants as constants

# TODO: consider adding filling with liquid and being used up (don't need
# cookable since being cooked is very similar to being heated, and we're
# tracking object states outside the environment anyways)
#
# canChangeTempToHot is also a property of Toaster, StoveBurner, CoffeeMachine,
# and Microwave. Only Potato is cookable
class InteractionReward(object):
    """
    Simple reward function for InteractionExploration that gives a reward for
    every interaction.
    """
    def __init__(self, env, rewards, reward_rotations=False,
            reward_look_angle=False, reward_state_changes=True):
        self.env = env
        self.rewards = rewards
        self.reward_rotations = reward_rotations
        self.reward_look_angle = reward_look_angle
        self.reset()

    def get_reward(self, state, action, api_success=True,
            target_instance_id=None, interact_mask=None):
        """
        state.metadata['lastAction'] has TeleportFull for navigation actions
        and is consistent with what was executed in the environment, while our
        action argument is consistent with the actions exposed to the
        agent/user.

        api_success is needed whether the THOR API rejected the action before
        the action was applied in the environment, since failed actions due to
        API rejection don't show up in last_action(state)'s metadata
        """
        reward = self.rewards['step_penalty']
        if not state.metadata['lastActionSuccess'] or not api_success:
            reward = self.rewards['invalid_action']
        elif (state.metadata['lastActionSuccess'] and
                state.metadata['lastAction'] in constants.INT_ACTIONS):
            interaction = (target_instance_id, state.metadata['lastAction'])
            if interaction not in self.interactions:
                self.interactions.append(interaction)
                reward = self.rewards['interaction']
            # Also, if action resulted in objects becoming clean, heated, or
            # cooled, give reward for each one
            newly_cleaned = (len(self.env.cleaned_objects) -
                    self.num_cleaned_objects)
            newly_heated = (len(self.env.heated_objects) -
                    self.num_heated_objects)
            newly_cooled = (len(self.env.cooled_objects) -
                    self.num_cooled_objects)
            # A single action can only result in one type of state change
            if newly_cleaned > 0:
                reward += newly_cleaned * self.rewards['state_change']
                self.num_cleaned_objects += newly_cleaned
            if newly_heated > 0:
                reward += newly_heated * self.rewards['state_change']
                self.num_heated_objects += newly_heated
            if newly_cooled > 0:
                reward += newly_cooled * self.rewards['state_change']
                self.num_cooled_objects += newly_cooled
        # Could also do state.metadata['lastAction'] == 'TeleportFull'
        elif (state.metadata['lastActionSuccess'] and action in
                constants.NAV_ACTIONS):
            location = state.pose_discrete[:2]
            if location not in self.visited_locations_poses:
                reward = self.rewards['navigation']
                self.visited_locations_poses[location] = [state.pose_discrete[2:]]
            elif location in self.visited_locations_poses:
                rotation = state.pose_discrete[2]
                look_angle = state.pose_discrete[3]
                if self.reward_rotations and len([rotation_look_angle for
                        rotation_look_angle in
                        self.visited_locations_poses[location] if
                        rotation_look_angle[0] == rotation]) == 0:
                    reward = self.rewards['navigation']
                    self.visited_locations_poses[location].append((rotation,
                        look_angle))
                elif self.reward_look_angle and len([rotation_look_angle for
                        rotation_look_angle in
                        self.visited_locations_poses[location] if
                        rotation_look_angle[1] == look_angle]) == 0:
                    reward = self.rewards['navigation']
                    self.visited_locations_poses[location].append((rotation,
                        look_angle))
                elif ((rotation, look_angle) not in
                        self.visited_locations_poses[location]):
                    self.visited_locations_poses[location].append((rotation,
                        look_angle))

        return reward

    def invalid_action(self):
        return self.rewards['invalid_action']

    def reset(self):
        self.interactions = [] # Tuples of object_id, action
        self.visited_locations_poses = {} # (x, z): (rotation, looking angle)
        # Use env/thor_env.py's cleaned_objects, heated_objects, and
        # cooled_objects, which don't ever drop members, to track
        # cleaned/heated/cooled state changes
        self.num_cleaned_objects = 0
        self.num_heated_objects = 0
        self.num_cooled_objects = 0
        # Do not reward initial agent location
        starting_pose = self.env.last_event.pose_discrete
        self.visited_locations_poses[starting_pose[:2]] = [starting_pose[2:]]

class BaseAction(object):
    '''
    base class for API actions
    '''

    def __init__(self, gt_graph, env, rewards, strict=True):
        self.gt_graph = gt_graph # for navigation
        self.env = env
        self.rewards = rewards
        self.strict = strict

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        reward, done = self.rewards['neutral'], True
        return reward, done


class GotoLocationAction(BaseAction):
    '''
    MoveAhead, Rotate, Lookup
    '''

    valid_actions = {'MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Teleport', 'TeleportFull'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        curr_pose = state.pose_discrete
        prev_pose = prev_state.pose_discrete
        tar_pose = tuple([int(i) for i in subgoal['location'].split('|')[1:]])

        prev_actions, _ = self.gt_graph.get_shortest_path(prev_pose, tar_pose)
        curr_actions, _ = self.gt_graph.get_shortest_path(curr_pose, tar_pose)

        prev_distance = len(prev_actions)
        curr_distance = len(curr_actions)
        reward = (prev_distance - curr_distance) * 0.2 # distance reward factor?

        # Consider navigation a success if we can see the target object in the next step from here.
        assert len(expert_plan) > goal_idx + 1
        next_subgoal = expert_plan[goal_idx + 1]['planner_action']
        next_goal_object = get_object(next_subgoal['objectId'], state.metadata)
        done = next_goal_object['visible'] and curr_distance < self.rewards['min_reach_distance']

        if done:
            reward += self.rewards['positive']

        return reward, done


class PickupObjectAction(BaseAction):
    '''
    PickupObject
    '''

    valid_actions = {'PickupObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        inventory_objects = state.metadata['inventoryObjects']
        if len(inventory_objects):
            inv_object_id = state.metadata['inventoryObjects'][0]['objectId']
            goal_object_id = subgoal['objectId']
            reward, done = (self.rewards['positive'], True) if inv_object_id == goal_object_id else (self.rewards['negative'], False)
        return reward, done


class PutObjectAction(BaseAction):
    '''
    PutObject
    '''

    valid_actions = {'PutObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_object_id = subgoal['objectId']
        recep_object = get_object(subgoal['receptacleObjectId'], state.metadata)
        if recep_object is not None:
            is_target_in_recep = target_object_id in recep_object['receptacleObjectIds']
            reward, done = (self.rewards['positive'], True) if is_target_in_recep else (self.rewards['negative'], False)
        return reward, done


class OpenObjectAction(BaseAction):
    '''
    OpenObject
    '''

    valid_actions = {'OpenObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_recep = get_object(subgoal['objectId'], state.metadata)
        if target_recep is not None:
            is_target_open = target_recep['isOpen']
            reward, done = (self.rewards['positive'], True) if is_target_open else (self.rewards['negative'], False)
        return reward, done


class CloseObjectAction(BaseAction):
    '''
    CloseObject
    '''

    valid_actions = {'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['negative'], False
        target_recep = get_object(subgoal['objectId'], state.metadata)
        if target_recep is not None:
            is_target_closed = not target_recep['isOpen']
            reward, done = (self.rewards['positive'], True) if is_target_closed else (self.rewards['negative'], False)
        return reward, done


class ToggleObjectAction(BaseAction):
    '''
    ToggleObjectOn, ToggleObjectOff
    '''

    valid_actions = {'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_toggle = get_object(subgoal['objectId'], state.metadata)
        if target_toggle is not None:
            is_target_toggled = target_toggle['isToggled']
            reward, done = (self.rewards['positive'], True) if is_target_toggled else (self.rewards['negative'], False)
        return reward, done


class SliceObjectAction(BaseAction):
    '''
    SliceObject
    '''

    valid_actions = {'SliceObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_object = get_object(subgoal['objectId'], state.metadata)
        if target_object is not None:
            is_target_sliced = target_object['isSliced']
            reward, done = (self.rewards['positive'], True) if is_target_sliced else (self.rewards['negative'], False)
        return reward, done


class CleanObjectAction(BaseAction):
    '''
    CleanObject
    '''

    valid_actions = {'PutObject', 'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        clean_object = get_object(subgoal['cleanObjectId'], state.metadata)
        if clean_object is not None:
            is_obj_clean = clean_object['objectId'] in self.env.cleaned_objects
            reward, done = (self.rewards['positive'], True) if is_obj_clean else (self.rewards['negative'], False)
        return reward, done


class HeatObjectAction(BaseAction):
    '''
    HeatObject
    '''

    valid_actions = {'OpenObject', 'CloseObject', 'PickupObject', 'PutObject', 'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        reward, done = self.rewards['neutral'], False
        next_put_goal_idx = goal_idx+2 # (+1) GotoLocation -> (+2) PutObject (get the objectId from the PutObject action)
        if next_put_goal_idx < len(expert_plan):
            heat_object_id = expert_plan[next_put_goal_idx]['planner_action']['objectId']
            heat_object = get_object(heat_object_id, state.metadata)
            is_obj_hot = heat_object['objectId'] in self.env.heated_objects
            reward, done = (self.rewards['positive'], True) if is_obj_hot else (self.rewards['negative'], False)
        return reward, done


class CoolObjectAction(BaseAction):
    '''
    CoolObject
    '''

    valid_actions = {'OpenObject', 'CloseObject', 'PickupObject', 'PutObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state.metadata['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        reward, done = self.rewards['neutral'], False
        next_put_goal_idx = goal_idx+2 # (+1) GotoLocation -> (+2) PutObject (get the objectId from the PutObject action)
        if next_put_goal_idx < len(expert_plan):
            cool_object_id = expert_plan[next_put_goal_idx]['planner_action']['objectId']
            cool_object = get_object(cool_object_id, state.metadata)
            is_obj_cool = cool_object['objectId'] in self.env.cooled_objects
            reward, done = (self.rewards['positive'], True) if is_obj_cool else (self.rewards['negative'], False)
        return reward, done


def get_action(action_type, gt_graph, env, reward_config, strict):
    action_type_str = action_type + "Action"

    if action_type_str in globals():
        action = globals()[action_type_str]
        return action(gt_graph, env, reward_config[action_type_str], strict)
    else:
        raise Exception("Invalid action_type %s" % action_type_str)
