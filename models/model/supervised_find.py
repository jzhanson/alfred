import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'data'))
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
from data.fo_dataset import get_dataloaders
from models.utils.metric import compute_actions_f1

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
parser.add_argument('-ud', '--use-dataset', dest='use_dataset', action='store_true')
parser.add_argument('-nud', '--no-use-dataset', dest='use_dataset', action='store_false')
parser.set_defaults(use_dataset=True)
parser.add_argument('-dt', '--dataset-transitions', dest='dataset_transitions', action='store_true')
parser.add_argument('-ndt', '--dataset-trajectories', dest='dataset_transitions', action='store_false')
parser.set_defaults(dataset_transitions=False)
parser.add_argument('-bs', '--batch-size', type=int, default=1, help='batch size of training trajectories or transitions if dataset-transitions is set')

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
# I got these splits out of the last number in the first directory of each
# train, valid_seen and valid_unseen task
TRAIN_SCENE_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 220, 221, 222, 223, 224, 225, 227, 228, 229, 230, 301, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 326, 327, 328, 329, 330, 401, 402, 403, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 426, 427, 428, 429, 430]
VALID_SEEN_SCENE_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 201, 202, 203, 204, 205, 206, 207, 212, 213, 214, 216, 218, 222, 223, 224, 225, 227, 229, 230, 301, 302, 303, 304, 305, 309, 310, 311, 313, 314, 316, 318, 320, 323, 324, 326, 327, 328, 329, 330, 401, 402, 403, 405, 406, 407, 408, 409, 410, 412, 413, 414, 415, 417, 418, 419, 422, 423, 426, 427, 428, 429]
VALID_UNSEEN_SCENE_NUMBERS = [10, 219, 308, 424]
TEST_SCENE_NUMBERS = [9, 29, 215, 226, 315, 325, 404, 425]

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

def flatten_trajectories(batch_samples, frame_stack=1,
        zero_fill_frame_stack=False):
    """
    Given batch_samples, a dict where keys are attributes of trajectories and
    the values are lists containing the attributes for each trajectory,
    assemble a dict where keys are attributes of transitions and the values are
    lists containing the attributes for each transition.

    I.e. batch_samples['target'] contains one element for every trajectory
    while batch_samples['images'] contains a list (one element for every
    transition) for every trajectory. flat_batch_samples['target'] contains one
    element for every transition, and flat_batch_samples['images'] contains one
    element for every transition.
    """
    flat_targets = []
    flat_actions = []
    for i in range(len(batch_samples['target'])):
        trajectory_actions = batch_samples['low_actions'][i]
        target = batch_samples['target'][i]
        # Repeat trajectory targets for each action
        flat_targets.extend([target for _ in trajectory_actions])
        flat_actions.extend([action for action in trajectory_actions])
    # Stack frames along channel dimension
    trajectory_images = []
    for i in range(len(batch_samples['images'])):
        for j in range(len(batch_samples['images'][i])):
            if j < frame_stack - 1:
                if zero_fill_frame_stack:
                    # Fill earlier frames with zeroes
                    frames = torch.cat([torch.zeros((frame_stack - j
                        - 1 * 3), 300, 300)] + [frame.permute(2, 0, 1) for
                            frame in batch_samples['images'][i][:j+1]])
                else:
                    # Repeat first frame
                    frames = torch.cat([batch_samples['images'][i][0].permute(
                        2, 0, 1) for _ in range(frame_stack
                                    - j - 1)] + [frame.permute(2, 0, 1) for
                                        frame in batch_samples['images'][i][
                                            :j+1]])
                trajectory_images.append(frames)
            else:
                trajectory_images.append(torch.cat([frame.permute(2, 0, 1) for
                    frame in batch_samples['images'][i][
                        j-frame_stack+1:j+1]]))

    flat_images = torch.stack(trajectory_images).to(device=device,
            dtype=torch.float32)

    flat_batch_samples = {}
    flat_batch_samples['target'] = flat_targets
    flat_batch_samples['low_actions'] = flat_actions
    flat_batch_samples['images'] = flat_images
    # TODO: do the same for features
    flat_batch_samples['features'] = []
    return flat_batch_samples

def actions_accuracy_f1(predicted_action_indexes, expert_action_indexes):
    """
    Calculate the accuracy and f1 (micro) of predicted actions.
    """
    correct_preds = 0
    for i in range(len(expert_action_indexes)):
        if predicted_action_indexes[i] == expert_action_indexes[i]:
            correct_preds += 1
    accuracy = correct_preds / len(predicted_action_indexes)
    f1 = compute_actions_f1(predicted_action_indexes,
            expert_action_indexes).item()
    return accuracy, f1

def train_dataset(fo, model, optimizer, dataloaders, obj_type_to_index,
        dataset_transitions=False, batch_size=1, frame_stack=1,
        zero_fill_frame_stack=False, eval_interval=100):
    """
    Train a model by sampling from a torch dataloader.

    dataloaders is a dict with 'train', 'valid_seen', 'valid_unseen' as keys
    and values of those dataloaders.
    """
    writer = SummaryWriter(log_dir='tensorboard_logs')
    train_iter = 0
    train_frames = 0
    train_trajectories = 0
    last_losses = []
    last_accuracies = []
    last_f1s = []

    while True:
        for batch_samples in dataloaders['train']:
            if dataset_transitions:
                # TODO: test the transitions branch of the code
                flat_targets = batch_samples['target']
                flat_actions = batch_samples['low_actions']
                # Stacking frames doesn't make sense
                flat_images = batch_samples['images']
                flat_features = batch_samples['features']
            else:
                flat_batch_samples = flatten_trajectories(
                        batch_samples, frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack)
                flat_targets = flat_batch_samples['target']
                flat_actions = flat_batch_samples['low_actions']
                flat_images = flat_batch_samples['images']
                flat_features = flat_batch_samples['features']

            # Turn target names into indexes and action names into indexes
            flat_target_indexes = torch.tensor([obj_type_to_index[
                constants.OBJECTS_LOWER_TO_UPPER[target]] for target in flat_targets],
                device=device)
            flat_action_indexes = torch.tensor([ACTION_TO_INDEX[action] for
                action in flat_actions], device=device)

            # Train
            action_scores = model(flat_images, flat_target_indexes)
            loss = F.cross_entropy(action_scores, flat_action_indexes)
            optimizer.zero_grad()
            # TODO: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
            #loss.backward()
            try:
                loss.backward(retain_graph=True)
            except:
                loss.backward()
            # TODO: may want to clamp gradients
            optimizer.step()

            train_iter += 1
            train_frames += len(action_scores)
            train_trajectories += batch_size
            last_losses.append(loss.item())
            accuracy, f1 = actions_accuracy_f1(torch.argmax(action_scores,
                dim=1), flat_action_indexes)
            last_accuracies.append(accuracy)
            last_f1s.append(f1)

            writer.add_scalar('train/loss', loss, train_iter)
            writer.add_scalar('train/loss_frames', loss, train_frames)
            writer.add_scalar('train/loss_trajectories', loss,
                    train_trajectories)
            writer.add_scalar('train/accuracy', accuracy, train_iter)
            writer.add_scalar('train/accuracy_frames', accuracy, train_frames)
            writer.add_scalar('train/accuracy_trajectories', accuracy,
                    train_trajectories)
            writer.add_scalar('train/f1', f1, train_iter)
            writer.add_scalar('train/f1_frames', f1, train_frames)
            writer.add_scalar('train/f1_trajectories', f1,
                    train_trajectories)
            # Record average policy entropy over all trajectories/transitions
            with torch.no_grad():
                entropy = trajectory_avg_entropy(action_scores)
            writer.add_scalar('train/entropy', entropy, train_iter)
            writer.add_scalar('train/entropy_frames', entropy, train_frames)
            writer.add_scalar('train/entropy_trajectories', entropy,
                    train_trajectories)

            if train_iter % eval_interval == 0:
                last_losses_mean = np.mean(last_losses)
                writer.add_scalar('train/avg_loss', last_losses_mean,
                        train_iter)
                writer.add_scalar('train/avg_loss_frames', last_losses_mean,
                        train_frames)
                writer.add_scalar('train/avg_loss_trajectories',
                        last_losses_mean, train_trajectories)
                last_accuracies_mean = np.mean(last_accuracies)
                writer.add_scalar('train/avg_accuracy', last_accuracies_mean,
                        train_iter)
                writer.add_scalar('train/avg_accuracy_frames',
                        last_accuracies_mean, train_frames)
                writer.add_scalar('train/avg_accuracy_trajectories',
                        last_accuracies_mean, train_trajectories)
                last_f1s_mean = np.mean(last_f1s)
                writer.add_scalar('train/avg_f1', last_f1s_mean, train_iter)
                writer.add_scalar('train/avg_f1_frames', last_f1s_mean,
                        train_frames)
                writer.add_scalar('train/avg_f1_trajectories', last_f1s_mean,
                        train_trajectories)

                print('iteration %d frames %d trajectories %d avg loss %.6f \
                        avg accuracy %.6f avg f1 %.6f' % (train_iter,
                            train_frames, train_trajectories, last_losses_mean,
                            last_accuracies_mean, last_f1s_mean))

                last_losses = []
                last_accuracies = []
                last_f1s = []

                # Evaluate on valid_seen and valid_unseen
                results = eval_dataset(model, dataloaders, obj_type_to_index,
                        dataset_transitions=dataset_transitions,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack)
                for split in ['valid_seen', 'valid_unseen']:
                    writer.add_scalar(split + '/accuracy',
                            results[split]['accuracy'], train_iter)
                    writer.add_scalar(split + '/accuracy_frames',
                            results[split]['accuracy'], train_frames)
                    writer.add_scalar(split + '/accuracy_trajectories',
                            results[split]['accuracy'], train_trajectories)
                    writer.add_scalar(split + '/f1',
                            results[split]['f1'], train_iter)
                    writer.add_scalar(split + '/f1_frames',
                            results[split]['f1'], train_frames)
                    writer.add_scalar(split + '/f1_trajectories',
                            results[split]['f1'], train_trajectories)
                    print(split + ' accuracy %.6f f1 %.6f' %
                            (results[split]['accuracy'],
                                results[split]['f1']))

                seen_results, unseen_results = eval_online(fo, model,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        seen_episodes=10, unseen_episodes=10)
                write_eval_results(writer, seen_results, unseen_results,
                        train_iter, train_frames, train_trajectories)

def eval_dataset(model, dataloaders, obj_type_to_index,
        dataset_transitions=False,frame_stack=1, zero_fill_frame_stack=False):
    """
    Evaluate a model on valid_seen and valid_unseen datasets given by
    dataloaders.
    """
    results = {}

    model.eval()
    for split in ['valid_seen', 'valid_unseen']:
        results[split] = {}
        all_predicted_action_indexes = []
        all_expert_action_indexes = []
        for batch_samples in dataloaders[split]:
            if dataset_transitions:
                # TODO: test the transitions branch of the code
                flat_targets = batch_samples['target']
                flat_actions = batch_samples['low_actions']
                # Stacking frames doesn't make sense
                flat_images = batch_samples['images']
                flat_features = batch_samples['features']
            else:
                flat_batch_samples = flatten_trajectories(batch_samples,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack)
                flat_targets = flat_batch_samples['target']
                flat_actions = flat_batch_samples['low_actions']
                flat_images = flat_batch_samples['images']
                flat_features = flat_batch_samples['features']

            # Turn target names into indexes and action names into indexes
            flat_target_indexes = torch.tensor([obj_type_to_index[
                constants.OBJECTS_LOWER_TO_UPPER[target]] for target in
                flat_targets], device=device)
            flat_action_indexes = torch.tensor([ACTION_TO_INDEX[action] for
                action in flat_actions], device=device)

            with torch.no_grad():
                action_scores = model(flat_images, flat_target_indexes)
            all_predicted_action_indexes.append(torch.argmax(action_scores,
                dim=1))
            all_expert_action_indexes.append(flat_action_indexes)

        all_predicted_action_indexes = torch.cat(all_predicted_action_indexes)
        all_expert_action_indexes = torch.cat(all_expert_action_indexes)

        accuracy, f1 = actions_accuracy_f1(all_predicted_action_indexes,
                all_expert_action_indexes)
        results[split]['accuracy'] = accuracy
        results[split]['f1'] = f1

    model.train()
    return results


def train(fo, model, optimizer, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, eval_interval=1000):
    """
    Train a model by collecting a trajectory online, then training with correct
    action supervision.
    """
    writer = SummaryWriter(log_dir='tensorboard_logs')
    train_iter = 0
    train_frames = 0
    last_losses = []
    last_successes = [] # tuples of (success, path_weighted_success)
    # tuples of (crow_distance, walking_distance, actions_distance i.e. how
    # many actions left in expert trajectory)
    last_distances_to_goal = []
    last_trajectory_lengths = []
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
        last_trajectory_lengths.append(len(trajectory_results['frames']))

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
            writer.add_scalar('train/avg_loss_frames', last_losses_mean,
                    train_frames)

            last_successes_mean = np.mean([x[0] for x in last_successes])
            last_path_weighted_successes_mean = np.mean([x[1] for x in
                last_successes])
            writer.add_scalar('train/avg_success', last_successes_mean,
                    train_iter)
            writer.add_scalar('train/avg_success_frames', last_successes_mean,
                    train_frames)
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

            last_trajectory_lengths_mean = np.mean(last_trajectory_lengths)
            writer.add_scalar('train/avg_trajectory_length',
                    last_trajectory_lengths_mean, train_iter)
            writer.add_scalar('train/avg_trajectory_length_frames',
                    last_trajectory_lengths_mean, train_frames)

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

            # Collect validation statistics and write, print
            seen_results, unseen_results = eval_online(fo, model,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack)

            write_eval_results(writer, seen_results, unseen_results,
                    train_iter, train_frames)

# TODO: don't repeat so much tensorboard code
def write_eval_results(writer, seen_results, unseen_results, train_iter,
        train_frames, train_trajectories=None):

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
    if train_trajectories is not None:
        writer.add_scalar('validation/seen/avg_success_trajectories',
                seen_successes_mean, train_trajectories)
        writer.add_scalar(
                'validation/seen/avg_path_weighted_success_trajectories',
                seen_path_weighted_successes_mean, train_trajectories)
        writer.add_scalar('validation/unseen/avg_success_trajectories',
                unseen_successes_mean, train_trajectories)
        writer.add_scalar(
                'validation/unseen/avg_path_weighted_success_trajectories',
                unseen_path_weighted_successes_mean, train_trajectories)

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
    if train_trajectories is not None:
        writer.add_scalar('validation/seen/avg_actions_distance_trajectories',
                seen_actions_distances_mean, train_trajectories)
        writer.add_scalar(
                'validation/unseen/avg_actions_distance_trajectories',
                unseen_actions_distances_mean, train_trajectories)

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
    if train_trajectories is not None:
        writer.add_scalar(
                'validation/seen/avg_trajectory_entropy_trajectories',
                seen_entropys_mean, train_trajectories)
        writer.add_scalar(
                'validation/unseen/avg_trajectory_entropy_trajectories',
                unseen_entropys_mean, train_trajectories)

    # Mean over trajectories of mean entropy per trajectory
    seen_trajectory_lengths_mean = torch.mean(torch.tensor(
        seen_results['trajectory_lengths'], device=device))
    unseen_trajectory_lengths_mean = torch.mean(torch.tensor(
        unseen_results['entropys'], device=device))
    writer.add_scalar('validation/seen/avg_trajectory_length',
            seen_trajectory_lengths_mean, train_iter)
    writer.add_scalar('validation/seen/avg_trajectory_length_frames',
            seen_trajectory_lengths_mean, train_frames)
    writer.add_scalar('validation/unseen/avg_trajectory_length',
            unseen_trajectory_lengths_mean, train_iter)
    writer.add_scalar('validation/unseen/avg_trajectory_length_frames',
            unseen_trajectory_lengths_mean, train_frames)
    if train_trajectories is not None:
        writer.add_scalar(
                'validation/seen/avg_trajectory_length_trajectories',
                seen_trajectory_lengths_mean, train_trajectories)
        writer.add_scalar(
                'validation/unseen/avg_trajectory_length_trajectories',
                unseen_trajectory_lengths_mean, train_trajectories)


def eval_online(fo, model, frame_stack=1, zero_fill_frame_stack=False,
        seen_episodes=1, unseen_episodes=1):
    model.eval()
    successes = [] # tuples of (success, path_weighted_success)
    # tuples of (crow_distance, walking_distance, actions_distance i.e. how
    # many actions left in expert trajectory)
    distances_to_goal = []
    entropys = []
    trajectory_lengths = []
    # Evaluate on training (seen) scenes
    for i in range(seen_episodes):
        with torch.no_grad():
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
        trajectory_lengths.append(float(len(trajectory_results['frames'])))

    seen_results = {}
    seen_results['successes'] = successes
    seen_results['distances_to_goal'] = distances_to_goal
    seen_results['entropys'] = entropys
    seen_results['trajectory_lengths'] = trajectory_lengths

    # Evaluate on validation (unseen) scenes
    successes = []
    distances_to_goal = []
    entropys = []
    trajectory_lengths = []
    for i in range(unseen_episodes):
        with torch.no_grad():
            trajectory_results = rollout_trajectory(fo, model,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    scene_name_or_num=random.choice(VALID_UNSEEN_SCENE_NUMBERS))
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
        trajectory_lengths.append(float(len(trajectory_results['frames'])))
    unseen_results = {}
    unseen_results['successes'] = successes
    unseen_results['distances_to_goal'] = distances_to_goal
    unseen_results['entropys'] = entropys
    unseen_results['trajectory_lengths'] = trajectory_lengths

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

    if args.use_dataset:
        # Frame stacking models only make sense if you're sampling trajectories
        # from a dataset, not transitions
        if args.dataset_transitions and args.frame_stack > 1:
            args.frame_stack = 1
        dataloaders = get_dataloaders(batch_size=args.batch_size,
                transitions=args.dataset_transitions)
    model = NatureCNN(len(obj_type_to_index), NUM_ACTIONS,
            frame_stack=args.frame_stack,
            object_embedding_dim=args.object_embedding_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.use_dataset:
        train_dataset(fo, model, optimizer, dataloaders, obj_type_to_index,
                dataset_transitions=args.dataset_transitions,
                batch_size=args.batch_size, frame_stack=args.frame_stack,
                zero_fill_frame_stack=args.zero_fill_frame_stack,
                eval_interval=args.eval_interval)
    else:
        train(fo, model, optimizer, frame_stack=args.frame_stack,
                zero_fill_frame_stack=args.zero_fill_frame_stack,
                teacher_force=args.teacher_force,
                eval_interval=args.eval_interval)

