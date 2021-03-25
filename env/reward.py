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
    def __init__(self, env, rewards, reward_rotations_look_angles=False,
            reward_state_changes=True, persist_state=False,
            repeat_discount=0.0):
        self.env = env
        self.rewards = rewards
        self.reward_rotations_look_angles = reward_rotations_look_angles
        self.persist_state = persist_state
        self.repeat_discount = repeat_discount
        self.trajectories = 0
        self.reset(init=True)

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

        Assumes invalid_action reward is not 0 and that if decayed
        navigation/interaction reward is 0 that means that step_penalty should
        be given instead.
        """
        if not state.metadata['lastActionSuccess'] or not api_success:
            reward = self.rewards['invalid_action']
        elif (state.metadata['lastActionSuccess'] and
                state.metadata['lastAction'] in constants.INT_ACTIONS):
            interaction = (target_instance_id, state.metadata['lastAction'])
            if interaction not in self.interactions:
                self.interactions[interaction] = 1
            else:
                self.interactions[interaction] += 1
            reward = self.rewards['interaction'] * pow(self.repeat_discount,
                    self.interactions[interaction] - 1)
            # Also, if action resulted in objects becoming clean, heated, or
            # cooled, give reward for each one
            newly_cleaned = (len(self.env.cleaned_objects) -
                    len(self.trajectory_cleaned_objects))
            newly_heated = (len(self.env.heated_objects) -
                    len(self.trajectory_heated_objects))
            newly_cooled = (len(self.env.cooled_objects) -
                    len(self.trajectory_cooled_objects))
            for env_objects, trajectory_objects, memory_objects in zip(
                    [self.env.cleaned_objects, self.env.heated_objects,
                        self.env.cooled_objects],
                    [self.trajectory_cleaned_objects,
                        self.trajectory_heated_objects,
                        self.trajectory_cooled_objects],
                    [self.cleaned_objects, self.heated_objects,
                        self.cooled_objects]):
                num_new_objects = len(env_objects) - len(trajectory_objects)
                if num_new_objects > 0:
                    # Go through all objects marked as cleaned/heated/cooled in
                    # the environment, and if it is newly added, add it to our
                    # tracking set and our memory
                    for object_id in env_objects:
                        if object_id not in trajectory_objects:
                            trajectory_objects.update(object_id)
                            if object_id not in memory_objects:
                                memory_objects[object_id] = 1
                            else:
                                memory_objects[object_id] += 1
                            reward += self.rewards['state_change'] * pow(
                                    self.repeat_discount,
                                    memory_objects[object_id] - 1)
                    # A single action can only result in one type of state
                    # change
                    break
        # Could also do state.metadata['lastAction'] == 'TeleportFull'
        elif (state.metadata['lastActionSuccess'] and action in
                constants.NAV_ACTIONS):
            location = state.pose_discrete[:2]
            rotation = state.pose_discrete[2]
            look_angle = state.pose_discrete[3]
            # The times_visited pattern is a little ugly but used because a
            # self.visited_locations_poses being a dict of lists of poses makes
            # getting a specific location + pose awkward
            if location not in self.visited_locations_poses:
                self.visited_locations_poses[location] = {
                        (rotation, look_angle) : 1}
            elif location in self.visited_locations_poses:
                if ((rotation, look_angle) not in
                        self.visited_locations_poses[location]):
                    self.visited_locations_poses[location][(rotation,
                        look_angle)] = 1
                else:
                    # Update times_visited
                    self.visited_locations_poses[location][(rotation,
                        look_angle)] += 1

            # At this point, self.visited_locations_poses and its counts are up
            # to date
            if self.reward_rotations_look_angles:
                times_visited = (self.visited_locations_poses[location][pose] -
                        1)
            else:
                times_visited = sum(self.visited_locations_poses[location]
                        .values()) - 1

            reward = self.rewards['navigation'] * pow(self.repeat_discount,
                    times_visited)

        # This is a bit of a hack so that self.repeat_decay = 0.0 works as
        # expected to make repeated actions take a step penalty instead of 0
        # reward. Assumes rewards['invalid_action'] is not 0
        if reward == 0:
            reward = self.rewards['step_penalty']
        return reward

    def invalid_action(self):
        return self.rewards['invalid_action']

    def reset(self, init=False):
        """
        Call reset() before starting a new episode, after env has been set up.
        """
        if init or not self.persist_state:
            # (object_id, action) : times taken
            self.interactions = {}
            # Dict(tuple, Dict(tuple, int))
            # (x, z): ((rotation, looking angle) : times in position)
            self.visited_locations_poses = {}
            # Object id : times cleaned/heated/cooled
            self.cleaned_objects = {}
            self.heated_objects = {}
            self.cooled_objects = {}
            # Mark initial agent location as visited
            starting_pose = self.env.last_event.pose_discrete
            self.visited_locations_poses[starting_pose[:2]] = {
                    starting_pose[2:] : 1}
        # Keep a copy of env/thor_env.py's cleaned_objects, heated_objects, and
        # cooled_objects, which don't ever drop members, to track
        # cleaned/heated/cooled state changes within a trajectory
        self.trajectory_cleaned_objects = set()
        self.trajectory_heated_objects = set()
        self.trajectory_cooled_objects = set()
        self.trajectories += 1

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
