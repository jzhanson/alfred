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
from env.find_one import FindOne, index_all_items, ACTIONS, NUM_ACTIONS, \
        ACTION_TO_INDEX, ACTIONS_DONE
from models.nn.fo import NatureCNN
import gen.constants as constants
from gen.graph.graph_obj import Graph

from tensorboardX import SummaryWriter

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-lr', '--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-oed', '--object-embedding-dim', type=int, default=16, help='object embedding dim')
parser.add_argument('-fs', '--frame-stack', type=int, default=3, help='number of frames to stack')
parser.add_argument('-zffs', '--zero-fill-frame-stack', dest='zero_fill_frame_stack', action='store_true', help='fill frames with zeros when frame stacking on early steps')
parser.add_argument('-fffs', '--first-fill-frame-stack', dest='zero_fill_frame_stack', action='store_false', help='replicate first frame when frame stacking on early steps')
parser.set_defaults(zero_fill_frame_stack=False)
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

# TODO: clean up moving model to CUDA
device = torch.device('cuda:3')

def trajectory_avg_entropy(trajectory_logits):
    return -torch.mean(torch.sum(
            F.log_softmax(trajectory_logits, dim=-1) *
            torch.exp(F.log_softmax(trajectory_logits, dim=-1)),
            dim=-1), dim=-1)

def rollout_trajectory(fo, model, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, scene_name_or_num=None):
    frames = []
    all_action_scores = []
    pred_action_indexes = []
    expert_action_indexes = []
    frame, target_object_index = fo.reset(scene_name_or_num)
    done = False
    while not done:
        frames.append(frame)

        # Most recent frames are last/later channels
        if len(frames) < frame_stack:
            # A frame is shape (300, 300, 3). Use transpose instead of reshape
            # to avoid data reordering. Concatenate along channels dimension
            if zero_fill_frame_stack:
                # Fill earlier frames with zeroes
                stacked_frames = torch.cat([torch.zeros(3 * (frame_stack -
                    len(frames)), 300, 300, device=device)] + \
                            [torch.from_numpy(np.ascontiguousarray(f)
                                .transpose(2, 0, 1)).float().to(device) for f
                                in frames])
            else:
                # Repeat first frame
                stacked_frames = torch.cat([torch.from_numpy(
                    np.ascontiguousarray(frames[0]).transpose(2, 0,
                        1)).float().to(device) for i in range(frame_stack -
                            len(frames))] + \
                                    [torch.from_numpy(np.ascontiguousarray(f)
                                        .transpose(2, 0, 1)).float().to(device)
                                        for f in frames])
        else:
            stacked_frames = torch.cat([torch.from_numpy(
                np.ascontiguousarray(f).transpose(2, 0, 1)).float().to(device)
                for f in frames[(len(frames) - frame_stack):]])
        stacked_frames = torch.unsqueeze(stacked_frames, dim=0)

        action_scores = model(stacked_frames,
                torch.tensor([target_object_index], device=device))
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

def train(fo, model, optimizer, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, eval_interval=100):
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
        trajectory_results = rollout_trajectory(fo, model,
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
                teacher_force=teacher_force,
                scene_name_or_num=random.choice(TRAIN_SCENE_NUMBERS))
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
        writer.add_scalar('train/loss_frames', loss, train_frames)
        # Record average policy entropy over an episode
        with torch.no_grad():
            entropy = trajectory_avg_entropy(all_action_scores)
        writer.add_scalar('train/entropy', entropy, train_iter)
        writer.add_scalar('train/entropy_frames', entropy, train_frames)

        print('train_iter: ' + str(train_iter))
        # Evaluate and save checkpoint every N trajectories, collect/print stats
        if train_iter % eval_interval == 0:
            last_losses_mean = np.mean(last_losses)
            writer.add_scalar('train/avg_loss', last_losses_mean, train_iter)
            writer.add_scalar('train/avg_loss_frames', last_losses_mean, train_frames)

            last_successes_mean = np.mean([x[0] for x in last_successes])
            last_path_weighted_successes_mean = np.mean([x[1] for x in
                last_successes])
            writer.add_scalar('train/avg_success', last_successes_mean, train_iter)
            writer.add_scalar('train/avg_success_frames', last_successes_mean, train_frames)
            writer.add_scalar('train/avg_path_weighted_success',
                    last_path_weighted_successes_mean, train_iter)
            writer.add_scalar('train/avg_path_weighted_success_frames',
                    last_path_weighted_successes_mean, train_frames)

            last_crow_distances_mean = np.mean([x[0] for x in
                last_distances_to_goal])
            last_walking_distances_mean = np.mean([x[1] for x in
                last_distances_to_goal])
            last_actions_distances_mean = np.mean([x[2] for x in
                last_distances_to_goal])
            writer.add_scalar('train/avg_crow_distance',
                    last_crow_distances_mean, train_iter)
            writer.add_scalar('train/avg_crow_distance_frames',
                    last_crow_distances_mean, train_frames)
            writer.add_scalar('train/avg_walking_distance',
                    last_walking_distances_mean, train_iter)
            writer.add_scalar('train/avg_walking_distance_frames',
                    last_walking_distances_mean, train_frames)
            writer.add_scalar('train/avg_actions_distance',
                    last_actions_distances_mean, train_iter)
            writer.add_scalar('train/avg_actions_distance_frames',
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
            seen_results, unseen_results = test(fo, model,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack)

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
            writer.add_scalar('validation/seen/avg_success_frames',
                    seen_successes_mean, train_frames)
            writer.add_scalar('validation/seen/avg_path_weighted_success',
                    seen_path_weighted_successes_mean, train_iter)
            writer.add_scalar('validation/seen/avg_path_weighted_success_frames',
                    seen_path_weighted_successes_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_success',
                    unseen_successes_mean, train_iter)
            writer.add_scalar('validation/unseen/avg_success_frames',
                    unseen_successes_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_path_weighted_success',
                    unseen_path_weighted_successes_mean, train_iter)
            writer.add_scalar(
                    'validation/unseen/avg_path_weighted_success_frames',
                    unseen_path_weighted_successes_mean, train_frames)

            # Mean actions left before goal
            seen_actions_distances_mean = np.mean([x[2] for x in
                seen_results['distances_to_goal']])
            unseen_actions_distances_mean = np.mean([x[2] for x in
                unseen_results['distances_to_goal']])
            writer.add_scalar('validation/seen/avg_actions_distance',
                    seen_actions_distances_mean, train_iter)
            writer.add_scalar('validation/seen/avg_actions_distance_frames',
                    seen_actions_distances_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_actions_distance',
                    unseen_actions_distances_mean, train_iter)
            writer.add_scalar('validation/unseen/avg_actions_distance_frames',
                    unseen_actions_distances_mean, train_frames)

            # Mean over trajectories of mean entropy per trajectory
            seen_entropys_mean = torch.mean(torch.tensor(
                seen_results['entropys'], device=device))
            unseen_entropys_mean = torch.mean(torch.tensor(
                unseen_results['entropys'], device=device))
            writer.add_scalar('validation/seen/avg_trajectory_entropy',
                    seen_entropys_mean, train_iter)
            writer.add_scalar('validation/seen/avg_trajectory_entropy_frames',
                    seen_entropys_mean, train_frames)
            writer.add_scalar('validation/unseen/avg_trajectory_entropy',
                    unseen_entropys_mean, train_iter)
            writer.add_scalar('validation/unseen/avg_trajectory_entropy_frames',
                    unseen_entropys_mean, train_frames)

def test(fo, model, frame_stack=1, zero_fill_frame_stack=False,
        seen_episodes=1, unseen_episodes=1):
    model.eval()
    successes = [] # tuples of (success, path_weighted_success)
    # tuples of (crow_distance, walking_distance, actions_distance i.e. how
    # many actions left in expert trajectory)
    distances_to_goal = []
    entropys = []
    # Evaluate on training (seen) scenes
    for i in range(seen_episodes):
        trajectory_results = rollout_trajectory(fo, model,
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
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
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
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

    obj_type_to_index_path = os.path.join(os.environ['ALFRED_ROOT'], 'env',
            'obj_type_to_index.json')
    if not os.path.isfile(obj_type_to_index_path):
        obj_type_to_index = index_all_items(env)
        with open(obj_type_to_index_path, 'w') as jsonfile:
            json.dump(obj_type_to_index, jsonfile)
    else:
        with open(obj_type_to_index_path, 'r') as jsonfile:
            obj_type_to_index = json.load(jsonfile)
    print(obj_type_to_index)
    index_to_obj_type = {i: ot for ot, i in obj_type_to_index.items()}

    fo = FindOne(env, obj_type_to_index)

    model = NatureCNN(len(obj_type_to_index), NUM_ACTIONS,
            frame_stack=args.frame_stack,
            object_embedding_dim=args.object_embedding_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train(fo, model, optimizer, frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            teacher_force=args.teacher_force, eval_interval=args.eval_interval)

