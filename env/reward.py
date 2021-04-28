import numpy as np

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
            repeat_discount=0.0, use_novelty=False):
        self.env = env
        self.rewards = rewards
        self.reward_rotations_look_angles = reward_rotations_look_angles
        self.persist_state = persist_state
        self.repeat_discount = repeat_discount
        # If use_novelty=True, repeat_discount will not be used
        self.use_novelty = use_novelty
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

        Assumes if decayed navigation/interaction reward is 0 or ends up really
        small that means that step_penalty should be given instead.
        """
        if not state.metadata['lastActionSuccess'] or not api_success:
            if action in constants.NAV_ACTIONS:
                reward = self.rewards['invalid_navigation']
            elif (action in constants.INT_ACTIONS or action ==
                    constants.ACTIONS_INTERACT):
                reward = self.rewards['invalid_interaction']
        # Don't state.metadata['lastAction'] since ThorEnv often sets it to
        # weird things like 'TeleportFull' or 'Pass' (if a faucet is turned on)
        elif ((state.metadata['lastActionSuccess'] and api_success) and
                (action in constants.INT_ACTIONS or action ==
                    constants.ACTIONS_INTERACT)):
            interaction = (target_instance_id, state.metadata['lastAction'])
            if interaction not in self.interactions:
                self.interactions[interaction] = 1
            else:
                self.interactions[interaction] += 1
            if interaction not in self.trajectory_interactions:
                self.trajectory_interactions[interaction] = 1
            else:
                self.trajectory_interactions[interaction] += 1
            times_visited = self.interactions[interaction]
            reward = self.get_discounted_reward(self.rewards['interaction'],
                    times_visited)
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
                # TODO: cleaning/heating/cooling only gives reward once per
                # episode per object (which makes sense since you can only
                # clean/heat/cool objects once) while navigation and
                # interaction can give decayed reward multiple times per
                # episode. can we check cleaned/heated/cooled conditions in the
                # same way env/thor_env.py does and apply reward per
                # interaction?
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
                            times_visited = memory_objects[object_id]
                            reward += self.get_discounted_reward(
                                    self.rewards['state_change'],
                                    times_visited)
                    # A single action can only result in one type of state
                    # change
                    break
        # Could also do state.metadata['lastAction'] == 'TeleportFull'
        elif (state.metadata['lastActionSuccess'] and action in
                constants.NAV_ACTIONS):
            location = state.pose_discrete[:2]
            rotation = state.pose_discrete[2]
            look_angle = state.pose_discrete[3]
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
            if location not in self.trajectory_visited_locations_poses:
                self.trajectory_visited_locations_poses[location] = {(rotation,
                    look_angle) : 1}
            elif location in self.trajectory_visited_locations_poses:
                if ((rotation, look_angle) not in
                        self.trajectory_visited_locations_poses[location]):
                    self.trajectory_visited_locations_poses[location][
                            (rotation, look_angle)] = 1
                else:
                    # Update times_visited
                    self.trajectory_visited_locations_poses[location][
                            (rotation, look_angle)] += 1

            # At this point, self.visited_locations_poses and its counts are up
            # to date
            if self.reward_rotations_look_angles:
                times_visited = self.visited_locations_poses[location][pose]
                reward = self.get_discounted_reward(self.rewards['navigation'],
                        times_visited)
            else:
                times_visited = sum(self.visited_locations_poses[location]
                        .values())
                # Only give rewards for rotations if option is set (otherwise
                # rotating counts as visiting the same location again)
                if action == 'MoveAhead':
                    reward = self.get_discounted_reward(
                            self.rewards['navigation'], times_visited)
                else:
                    reward = 0

        # This is a bit of a hack so that self.repeat_discount = 0.0 works as
        # expected to make repeated actions take a step penalty instead of 0
        # reward and also so the model doesn't get a tiny tiny positive reward forever
        if (reward < 1e-5 and reward != self.rewards['invalid_navigation'] and
                reward != self.rewards['invalid_interaction']):
            reward = self.rewards['step_penalty']
        return reward

    def get_discounted_reward(self, raw_reward, times_visited):
        """
        times_visited is the up-to-date times visited, including the current
        instance.
        """
        if self.use_novelty:
            return raw_reward / np.sqrt(times_visited)
        else:
            # Subtract the current instance/time visited
            return raw_reward * pow(self.repeat_discount,
                    times_visited - 1)

    def invalid_action(self, interaction=False):
        if interaction:
            return self.rewards['invalid_interaction']
        else:
            return self.rewards['invalid_navigation']

    def reset(self, init=False, scene_name_or_num=None):
        """
        Call reset() before starting a new episode, after env has been set up.
        """
        self.scene_name_or_num = scene_name_or_num
        if init or not self.persist_state:
            # scene_num : (statistic : statistics)
            self.memory = {}
        if self.scene_name_or_num not in self.memory:
            self.memory[self.scene_name_or_num] = {}
            # (object_id, action) : times taken
            self.memory[self.scene_name_or_num]['interactions'] = {}
            # Dict(tuple, Dict(tuple, int))
            # (x, z): ((rotation, looking angle) : times in position)
            self.memory[self.scene_name_or_num]['visited_locations_poses'] = {}
            # Object id : times cleaned/heated/cooled
            self.memory[self.scene_name_or_num]['cleaned_objects'] = {}
            self.memory[self.scene_name_or_num]['heated_objects'] = {}
            self.memory[self.scene_name_or_num]['cooled_objects'] = {}
        # Put a reference to self.memory's dict in easier-to-access names and
        # also so we don't have to type out
        # self.memory[self.scene_name_or_num]['...'] all over the place
        self.interactions = self.memory[self.scene_name_or_num]['interactions']
        self.visited_locations_poses = self.memory[self.scene_name_or_num][
                'visited_locations_poses']
        self.cleaned_objects = self.memory[self.scene_name_or_num][
                'cleaned_objects']
        self.heated_objects = self.memory[self.scene_name_or_num][
                'heated_objects']
        self.cooled_objects = self.memory[self.scene_name_or_num][
                'cooled_objects']
        # Mark initial agent location as visited
        starting_pose = self.env.last_event.pose_discrete
        if starting_pose[:2] not in self.visited_locations_poses:
            self.visited_locations_poses[starting_pose[:2]] = {
                    starting_pose[2:] : 1}
        elif starting_pose[2:] not in self.visited_locations_poses[
                starting_pose[:2]]:
            self.visited_locations_poses[starting_pose[:2]][
                    starting_pose[2:]] = 1
        else:
            self.visited_locations_poses[starting_pose[:2]][
                    starting_pose[2:]] += 1

        # Keep a copy of env/thor_env.py's cleaned_objects, heated_objects, and
        # cooled_objects, which don't ever drop members, to track
        # cleaned/heated/cooled state changes within a trajectory
        self.trajectory_cleaned_objects = set()
        self.trajectory_heated_objects = set()
        self.trajectory_cooled_objects = set()
        # Also, keep track of interactions and visited_locations_poses for the
        # current trajectory for getting the coverage. Can't use self.memory,
        # and can't keep these and append to self.memory at the end because
        # computing the reward from self.memory and
        # self.trajectory_interactions would be complex
        self.trajectory_interactions = {}
        self.trajectory_visited_locations_poses = {}
        self.trajectories += 1

    def get_coverages(self):
        """
        Returns navigation only coverage, navigation + poses
        coverage, interaction coverage, state change coverage.
        """
        navigation_location_coverage = (len(
                self.trajectory_visited_locations_poses) /
                constants.SCENE_NAVIGATION_COVERAGES[self.scene_name_or_num])
        # 4 rotations, 7 look angles between -30 and 60 with increments of 15
        navigation_location_pose_coverage = (sum([len(poses) for poses in
            self.trajectory_visited_locations_poses.values()]) /
                (constants.SCENE_NAVIGATION_COVERAGES[self.scene_name_or_num] *
                        4 * 7))
        interaction_coverage = (len(self.trajectory_interactions) /
                constants.SCENE_INTERACTION_COVERAGES[self.scene_name_or_num])
        state_change_coverage = ((len(self.trajectory_cleaned_objects) +
                len(self.trajectory_heated_objects) +
                len(self.trajectory_cooled_objects)) /
                constants.SCENE_STATE_CHANGE_COVERAGES[self.scene_name_or_num])
        return (navigation_location_coverage,
                navigation_location_pose_coverage, interaction_coverage,
                state_change_coverage)

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
