import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import random
import json
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import cv2
from env.thor_env import ThorEnv
import gen.constants as constants
from gen.graph.graph_obj import Graph

from tensorboardX import SummaryWriter

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-lr', '--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-oed', '--object-embedding-dim', type=int, default=16, help='object embedding dim')
parser.add_argument('-fs', '--frame_stack', type=int, default=3, help='number of frames to stack')
parser.add_argument('-ei', '--eval-interval', type=int, default=100, help='number of training trajectories between evaluation trajectories')
parser.add_argument('-tf', '--teacher-force', dest='teacher_force', action='store_true')
parser.add_argument('-ntf', '--no-teacher-force', dest='teacher_force', action='store_false')
parser.set_defaults(teacher_force=False)

'''
parser.add_argument('-do', '--dropout', type=float, default=0.02, help='dropout prob')
parser.add_argument('-n', '--n-epochs', type=float, default=10, help='training epochs')
parser.add_argument('-sn', '--save-name', type=str, default='model', help='model save name')
parser.add_argument('-id', '--model-id', type=str, default='model', help='model id')
'''


# NOTE: THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
# is due to CUDA 9.0 instead of a more advanced CUDA
# So is RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
# TODO: upgrade CUDA in ALFRED docker container

# Available scenes are [1, 30], [201, 230], [301, 330], and [401, 430]
# Tragically this is hardcoded in ai2thor 2.1.0 in
# ai2thor/controller.py line 429
TRAIN_SCENE_NUMBERS = list(range(1, 21)) + list(range(201, 221)) + \
    list(range(301, 321)) + list(range(401, 421)) # 20 scenes per room type
VALIDATION_SCENE_NUMBERS = list(range(21, 26)) + list(range(221, 226)) + \
    list(range(321, 326)) + list(range(421, 426)) # 5 scenes per room type
TEST_SCENE_NUMBERS = list(range(26, 31)) + list(range(226, 231)) + \
    list(range(326, 331)) + list(range(426, 431)) # 5 scenes per room type


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

device = torch.device('cuda:3')

def index_all_items(env):
    """Iterates through each scene and puts all item types into a dict"""
    obj_type_to_index = {}
    for scene_num in TRAIN_SCENE_NUMBERS:
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
            scene_name_or_num = random.choice(TRAIN_SCENE_NUMBERS)
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


class NatureCNN(nn.Module):
    def __init__(self, num_objects, num_actions, frame_stack=3, object_embedding_dim=16,
            frame_width=300, frame_height=300):
        super(NatureCNN, self).__init__()
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.frame_stack = frame_stack
        self.object_embedding_dim = object_embedding_dim

        # From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=8, stride=4):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(frame_width,
            kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3,
            stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(frame_height,
            kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3,
            stride=1)
        linear_input_size = convw * convh * 64 + self.object_embedding_dim

        self.object_embedding = nn.Embedding(num_embeddings=self.num_objects,
                embedding_dim=self.object_embedding_dim)

        # Nature architecture with batchnorm
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3*self.frame_stack,
            out_channels=32, kernel_size=8, stride=4, bias=True),
            nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64,
            kernel_size=4, stride=2, bias=True),
            nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(64), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(in_features=linear_input_size,
            out_features=512, bias=True), nn.ReLU())
        self.action_logits = nn.Sequential(nn.Linear(in_features=512,
            out_features=NUM_ACTIONS, bias=True))


        # Layers declaration - model inspired by "Learning About Objects by
        # Learning to Interact with Them", https://arxiv.org/abs/2006.09306
        ''' self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
        out_channels=32,
            kernel_size=5, stride=3, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2,
            stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2,
            stride=2, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5,
            padding=2, stride=2, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        '''


    def forward(self, frames, obj_index):
        x = self.conv1(frames)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)

        # "Late Fusion"
        object_embed = self.object_embedding(torch.tensor(obj_index,
            device=device))
        # Reshape conv output to (N, -1) and concatenate object one hots
        x3 = self.fc1(torch.cat([x2.view(x2.size(0), -1), object_embed], -1))
        action_probs = self.action_logits(x3)

        return action_probs

def trajectory_avg_entropy(trajectory_logits):
    return -torch.mean(torch.sum(
            F.log_softmax(trajectory_logits, dim=-1) *
            torch.exp(F.log_softmax(trajectory_logits, dim=-1)),
            dim=-1), dim=-1)

def rollout_trajectory(fo, model, teacher_force=False, scene_name_or_num=None):
    frames = []
    all_action_scores = []
    pred_action_indexes = []
    expert_action_indexes = []
    frame, target_object_index = fo.reset(scene_name_or_num)
    done = False
    while not done:
        frames.append(frame)

        # Most recent frames are last/later channels
        if len(frames) < model.frame_stack:
            stacked_frames = [torch.zeros(1, 3 * (model.frame_stack -
                len(frames)), 300, 300, device=device)] + \
            [torch.from_numpy(np.ascontiguousarray(f) .reshape(1, 3, 300,
                300)).float().to(device) for f in
                frames]
        else:
            stacked_frames = [torch.from_numpy(np.ascontiguousarray(f)
                .reshape(1, 3, 300, 300)).float().to(device) for f in
                frames[(len(frames) - model.frame_stack):]]
        # Concatenate along channels dimension
        stacked_frames = torch.cat(stacked_frames, dim=1)

        action_scores = model(stacked_frames, [target_object_index])
        # Sorted in increasing order (rightmost is highest scoring action)
        sorted_scores, top_indices = torch.sort(action_scores)
        top_indices = top_indices.flatten()
        # Try each action until success
        pred_action_index = None
        for i in range(NUM_ACTIONS):
            pred_action_index = top_indices[i]
            if teacher_force:
                current_expert_actions, _ = fo.get_current_expert_actions_path()
                selected_action = current_expert_actions[0]['action']
            else:
                selected_action = ACTIONS[pred_action_index]
            (frame, _), _, done, (action_success, event, expert_action) = \
                    fo.step(selected_action)
            if action_success:
                # TODO: consider penalizing failed actions more
                break
        assert pred_action_index is not None
        all_action_scores.append(action_scores)
        pred_action_indexes.append(pred_action_index)
        expert_action_indexes.append(ACTION_TO_INDEX[expert_action])
        # Episode was successful if agent predicted ACTIONS_DONE (ending the
        # episode) and the expert action is also ACTIONS_DONE
        #
        # success is less meaningful if teacher forcing
        if pred_action_index == ACTION_TO_INDEX[ACTIONS_DONE] and \
                pred_action_index == ACTION_TO_INDEX[expert_action]:
            success = True
        else:
            success = False
    print('trajectory len: ' + str(len(all_action_scores)))

    trajectory_results = {}
    trajectory_results['frames'] = frames
    trajectory_results['all_action_scores'] = all_action_scores
    trajectory_results['pred_action_indexes'] = pred_action_indexes
    trajectory_results['expert_action_indexes'] = expert_action_indexes
    trajectory_results['success'] = success
    trajectory_results['expert_actions'] = fo.original_expert_actions
    return trajectory_results

def train(fo, model, optimizer, teacher_force=False, eval_interval=100):
    writer = SummaryWriter(log_dir='tensorboard_logs')
    train_iter = 0
    train_frames = 0
    last_losses = []
    last_successes = [] # tuples of (success, path_weighted_success)
    # tuples of (crow_distance, walking_distance, actions_distance i.e. how
    # many actions left in expert trajectory)
    last_distances_to_goal = []
    # TODO: want a replay memory?
    while True:
        # Collect a trajectory
        trajectory_results = rollout_trajectory(fo, model, teacher_force)
        all_action_scores = torch.cat(trajectory_results['all_action_scores'])

        # Train on trajectory
        loss = F.cross_entropy(all_action_scores,
                torch.tensor(trajectory_results['expert_action_indexes'],
                    device=device))
        optimizer.zero_grad()
        # TODO: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
        #loss.backward()
        try:
            loss.backward(retain_graph=True)
        except:
            loss.backward()
        # TODO: may want to clamp gradients
        optimizer.step()

        # Compute and save some stats
        train_iter += 1
        train_frames += len(trajectory_results['frames'])
        last_losses.append(loss.item())
        last_successes.append((float(trajectory_results['success']),
                float(trajectory_results['success']) *
                len(fo.original_expert_actions) /
                max(len(trajectory_results['frames']),
                    len(fo.original_expert_actions))))
        final_expert_actions, final_expert_path = \
                fo.get_current_expert_actions_path()
        last_distances_to_goal.append((fo.crow_distance_to_goal(),
            fo.walking_distance_to_goal(), len(final_expert_actions)))
        writer.add_scalar('train/loss', loss, train_iter)
        writer.add_scalar('train/loss_step', loss, train_frames)
        # Record average policy entropy over an episode
        entropy = trajectory_avg_entropy(all_action_scores)
        writer.add_scalar('train/entropy', entropy, train_iter)
        writer.add_scalar('train/entropy_step', entropy, train_frames)

        print('train_iter: ' + str(train_iter))
        # Evaluate and save checkpoint every N trajectories, collect/print stats
        if train_iter % eval_interval == 0:
            last_losses_mean = np.mean(last_losses)
            writer.add_scalar('train/avg_loss', last_losses_mean, train_iter)
            writer.add_scalar('train/avg_loss_step', last_losses_mean, train_frames)

            last_successes_mean = np.mean([x[0] for x in last_successes])
            last_path_weighted_successes_mean = np.mean([x[1] for x in
                last_successes])
            writer.add_scalar('train/avg_success', last_successes_mean, train_iter)
            writer.add_scalar('train/avg_success_step', last_successes_mean, train_frames)
            writer.add_scalar('train/avg_path_weighted_success',
                    last_path_weighted_successes_mean, train_iter)
            writer.add_scalar('train/avg_path_weighted_success_step',
                    last_path_weighted_successes_mean, train_frames)

            last_crow_distances_mean = np.mean([x[0] for x in
                last_distances_to_goal])
            last_walking_distances_mean = np.mean([x[1] for x in
                last_distances_to_goal])
            last_actions_distances_mean = np.mean([x[2] for x in
                last_distances_to_goal])
            writer.add_scalar('train/avg_crow_distance',
                    last_crow_distances_mean, train_iter)
            writer.add_scalar('train/avg_crow_distance_step',
                    last_crow_distances_mean, train_frames)
            writer.add_scalar('train/avg_walking_distance',
                    last_walking_distances_mean, train_iter)
            writer.add_scalar('train/avg_walking_distance_step',
                    last_walking_distances_mean, train_frames)
            writer.add_scalar('train/avg_actions_distance',
                    last_actions_distances_mean, train_iter)
            writer.add_scalar('train/avg_actions_distance_step',
                    last_actions_distances_mean, train_frames)

            print('iteration %d frames %d avg loss %.6f' % (train_iter,
                train_frames, last_losses_mean))
            print('avg success %.6f avg path weighted success %.6f' %
                    (last_successes_mean, last_path_weighted_successes_mean))
            print('avg walking distance %.6f avg walking distance %.6f avg \
                    actions distance %.6f' % (last_crow_distances_mean,
                        last_walking_distances_mean,
                        last_actions_distances_mean))

            last_losses = []
            last_successes = []
            last_distances_to_goal = []

            # Collect test statistics and write, print
            seen_results, unseen_results = test(fo, model)

            seen_successes_mean = np.mean([x[0] for x in
                seen_results['successes']])
            seen_path_weighted_successes_mean = np.mean([x[1] for x in
                seen_results['successes']])
            unseen_successes_mean = np.mean([x[0] for x in
                unseen_results['successes']])
            unseen_path_weighted_successes_mean = np.mean([x[1] for x in
                unseen_results['successes']])
            writer.add_scalar('validation/seen/avg_success',
                    seen_successes_mean, train_iter)
            writer.add_scalar('validation/seen/avg_success_step',
                    seen_successes_mean, train_frames)
            writer.add_scalar('validation/seen/avg_path_weighted_success',
                    seen_path_weighted_successes_mean, train_iter)
            writer.add_scalar('validation/seen/avg_path_weighted_success_step',
                    seen_path_weighted_successes_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_success',
                    unseen_successes_mean, train_iter)
            writer.add_scalar('validation/unseen/avg_success_step',
                    unseen_successes_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_path_weighted_success',
                    unseen_path_weighted_successes_mean, train_iter)
            writer.add_scalar(
                    'validation/unseen/avg_path_weighted_success_step',
                    unseen_path_weighted_successes_mean, train_frames)

            # Mean actions left before goal
            seen_actions_distances_mean = np.mean([x[2] for x in
                seen_results['distances_to_goal']])
            unseen_actions_distances_mean = np.mean([x[2] for x in
                unseen_results['distances_to_goal']])
            writer.add_scalar('validation/seen/avg_actions_distance',
                    seen_actions_distances_mean, train_iter)
            writer.add_scalar('validation/seen/avg_actions_distance_step',
                    seen_actions_distances_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_actions_distance',
                    unseen_actions_distances_mean, train_iter)
            writer.add_scalar('validation/unseen/avg_actions_distance_step',
                    unseen_actions_distances_mean, train_frames)

            # Mean over trajectories of mean entropy per trajectory
            seen_entropys_mean = torch.mean(seen_results['entropys'])
            unseen_entropys_mean = torch.mean(unseen_results['entropys'])
            writer.add_scalar('validation/seen/avg_trajectory_entropy',
                    seen_entropys_mean, train_iter)
            writer.add_scalar('validation/seen/avg_trajectory_entropy_step',
                    seen_entropys_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_trajectory_entropy',
                    unseen_entropys_mean, train_iter)
            writer.add_scalar('validation/unseen/avg_trajectory_entropy_step',
                    unseen_entropys_mean, train_frames)

def test(fo, model, seen_episodes=10, unseen_episodes=10):
    model.eval()
    successes = [] # tuples of (success, path_weighted_success)
    # tuples of (crow_distance, walking_distance, actions_distance i.e. how
    # many actions left in expert trajectory)
    distances_to_goal = []
    entropys = []
    # Evaluate on training (seen) scenes
    for i in range(seen_episodes):
        trajectory_results = rollout_trajectory(fo, model,
                scene_name_or_num=random.choice(TRAIN_SCENE_NUMBERS))
        successes.append((float(trajectory_results['success']),
                float(trajectory_results['success']) *
                len(fo.original_expert_actions) /
                max(len(trajectory_results['frames']),
                    len(fo.original_expert_actions))))
        final_expert_actions, final_expert_path = \
                fo.get_current_expert_actions_path()
        distances_to_goal.append((fo.crow_distance_to_goal(),
            fo.walking_distance_to_goal(), len(final_expert_actions)))
        entropys.append(trajectory_avg_entropy(torch.cat(
            trajectory_results['all_action_scores'])))

    seen_results = {}
    seen_results['successes'] = successes
    seen_results['distances_to_goal'] = distances_to_goal
    seen_results['entropys'] = entropys

    # Evaluate on validation (unseen) scenes
    successes = []
    distances_to_goal = []
    entropys = []
    for i in range(unseen_episodes):
        trajectory_results = rollout_trajectory(fo, model,
                scene_name_or_num=random.choice(VALIDATION_SCENE_NUMBERS))
        successes.append((float(trajectory_results['success']),
                float(trajectory_results['success']) *
                len(fo.original_expert_actions) /
                max(len(trajectory_results['frames']),
                    len(fo.original_expert_actions))))
        final_expert_actions, final_expert_path = \
                fo.get_current_expert_actions_path()
        distances_to_goal.append((fo.crow_distance_to_goal(),
            fo.walking_distance_to_goal(), len(final_expert_actions)))
        entropys.append(trajectory_avg_entropy(torch.cat(
            trajectory_results['all_action_scores'])))
    unseen_results = {}
    unseen_results['successes'] = successes
    unseen_results['distances_to_goal'] = distances_to_goal
    unseen_results['entropys'] = entropys

    model.train()

    return seen_results, unseen_results

if __name__ == '__main__':
    args = parser.parse_args()

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

    # TODO: clean up moving model to CUDA
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = NatureCNN(len(obj_type_to_index), NUM_ACTIONS,
            frame_stack=args.frame_stack,
            object_embedding_dim=args.object_embedding_dim).to(device)
    #model.to(torch.device('cuda:3'))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train(fo, model, optimizer, teacher_force=args.teacher_force)

