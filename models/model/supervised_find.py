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
parser.add_argument('-ees', '--eval-episodes-seen', type=int, default=10, help='number of episodes to evaluate live on seen scenes')
parser.add_argument('-eeu', '--eval-episodes-unseen', type=int, default=10, help='number of episodes to evaluate live on unseen scenes')
parser.add_argument('-tf', '--teacher-force', dest='teacher_force', action='store_true')
parser.add_argument('-ntf', '--no-teacher-force', dest='teacher_force', action='store_false')
parser.set_defaults(teacher_force=False)
parser.add_argument('-dp', '--dataset-path', type=str, default=None, help='path (directory) to dataset indexes of trajectories and obj_type_to_index, if using')
parser.add_argument('-dt', '--dataset-transitions', dest='dataset_transitions', action='store_true')
parser.add_argument('-ndt', '--dataset-trajectories', dest='dataset_transitions', action='store_false')
parser.set_defaults(dataset_transitions=False)
parser.add_argument('-bs', '--batch-size', type=int, default=1, help='batch size of training trajectories or transitions if dataset-transitions is set')
parser.add_argument('-sp', '--save-path', type=str, default=None, help='path (directory) to save models and tensorboard stats')
parser.add_argument('-si', '--save-intermediate', dest='save_intermediate', action='store_true', help='save intermediate checkpoints (once per eval interval)')
parser.add_argument('-nsi', '--no-save-intermediate', dest='save_intermediate', action='store_false', help='don\'t save intermediate checkpoints (once per eval interval)')
parser.set_defaults(save_intermediate=False)
parser.add_argument('-lp', '--load-path', type=str, default=None, help='path (.pth) to load model checkpoint from')

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

def path_weighted_success(success, num_agent_actions, num_expert_actions):
    return float(success) * num_expert_actions / max(num_agent_actions,
            num_expert_actions)

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
        zero_fill_frame_stack=False, eval_episodes_seen=10,
        eval_episodes_unseen=10, eval_interval=100, save_path=None,
        save_intermediate=False, load_path=None):
    """
    Train a model by sampling from a torch dataloader.

    dataloaders is a dict with 'train', 'valid_seen', 'valid_unseen' as keys
    and values of those dataloaders.
    """
    writer = SummaryWriter(log_dir='tensorboard_logs' if save_path is None
            else os.path.join(save_path, 'tensorboard_logs'))
    if load_path is not None:
        checkpoint = torch.load(load_path)
        train_iter = checkpoint['train_iter']
        train_frames = checkpoint['train_frames']
        train_trajectories = checkpoint['train_trajectories']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading from ' + load_path + ' iteration ' + str(train_iter))
    else:
        train_iter = 0
        train_frames = 0
        train_trajectories = 0
    last_metrics = {}
    last_metrics['loss'] = []
    last_metrics['accuracy'] = []
    last_metrics['f1'] = []
    last_metrics['entropy'] = []

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
            last_metrics['loss'].append(loss.item())
            accuracy, f1 = actions_accuracy_f1(torch.argmax(action_scores,
                dim=1), flat_action_indexes)
            last_metrics['accuracy'].append(accuracy)
            last_metrics['f1'].append(f1)
            # Record average policy entropy over all trajectories/transitions
            with torch.no_grad():
                entropy = trajectory_avg_entropy(action_scores)
            last_metrics['entropy'].append(entropy.item())

            for metric in last_metrics.keys():
                writer.add_scalar('train/' + metric, last_metrics[metric][-1],
                        train_iter)
                writer.add_scalar('train/frames_' + metric, last_metrics[metric][-1],
                        train_frames)
                writer.add_scalar('train/trajectories_' + metric,
                        last_metrics[metric][-1], train_trajectories)

            if train_iter % eval_interval == 0:
                print('iteration %d frames %d trajectories %d' % (train_iter,
                    train_frames, train_trajectories))
                for metric, values in last_metrics.items():
                    mean = np.mean(values)
                    writer.add_scalar('train_avg/' + metric, mean, train_iter)
                    writer.add_scalar('train_avg/frames_' + metric, mean,
                            train_frames)
                    writer.add_scalar('train_avg/trajectories_' + metric, mean,
                            train_trajectories)
                    print('avg ' + metric + ' %.6f' % mean)
                    last_metrics[metric] = []

                # Evaluate on valid_seen and valid_unseen
                results_dataset = eval_dataset(model, dataloaders,
                        obj_type_to_index,
                        dataset_transitions=dataset_transitions,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack)
                for split in ['valid_seen', 'valid_unseen']:
                    for metric in results_dataset[split].keys():
                        writer.add_scalar(split + '/' + metric,
                                results_dataset[split][metric], train_iter)
                        writer.add_scalar(split + '/frames_' + metric,
                                results_dataset[split][metric], train_frames)
                        writer.add_scalar(split + '/trajectories_' + metric,
                                results_dataset[split][metric], train_trajectories)
                    print(split + ' accuracy %.6f f1 %.6f' %
                            (results_dataset[split]['accuracy'],
                                results_dataset[split]['f1']))

                results_online = eval_online(fo, model,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        seen_episodes=eval_episodes_seen,
                        unseen_episodes=eval_episodes_unseen)
                write_eval_results(writer, results_online, train_iter, train_frames,
                        train_trajectories)

                if save_path is not None:
                    if save_intermediate:
                        checkpoint_save_path = os.path.join(save_path, 'model_' +
                                str(train_iter) + '.pth')
                    else:
                        checkpoint_save_path = os.path.join(save_path, 'model.pth')
                    print('saving to ' + checkpoint_save_path)
                    torch.save({
                        'train_iter' : train_iter,
                        'train_frames' : train_frames,
                        'train_trajectories' : train_trajectories,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                    }, checkpoint_save_path)

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
        teacher_force=False, eval_episodes_seen=10, eval_episodes_unseen=10,
        eval_interval=1000, save_path=None, save_intermediate=False,
        load_path=None):
    """
    Train a model by collecting a trajectory online, then training with correct
    action supervision. Loads model from checkpoint if load_path is not None.
    """
    writer = SummaryWriter(log_dir='tensorboard_logs' if save_path is None
            else os.path.join(save_path, 'tensorboard_logs'))
    if load_path is not None:
        checkpoint = torch.load(load_path)
        train_iter = checkpoint['train_iter']
        train_frames = checkpoint['train_frames']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading from ' + load_path + ' iteration ' + str(train_iter))
    else:
        train_iter = 0
        train_frames = 0
    # TODO: save/load metrics instead of relying on tensorboard and make metric
    # loading work well with tensorboard
    last_metrics = {}
    last_metrics['loss'] = []
    last_metrics['success'] = []
    last_metrics['path_weighted_success'] = []
    last_metrics['crow_distance_to_goal'] = []
    last_metrics['walking_distance_to_goal'] = []
    last_metrics['action_distance_to_goal'] = []
    last_metrics['trajectory_length'] = []
    last_metrics['entropy'] = []

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
        last_metrics['loss'].append(loss.item())
        last_metrics['success'].append(float(trajectory_results['success']))
        last_metrics['path_weighted_success'].append(
                path_weighted_success(
                    trajectory_results['success'],
                    len(trajectory_results['frames']),
                    len(fo.original_expert_actions)))

        final_expert_actions, final_expert_path = \
                fo.get_current_expert_actions_path()
        last_metrics['crow_distance_to_goal'].append(
                fo.crow_distance_to_goal())
        last_metrics['walking_distance_to_goal'].append(
                fo.walking_distance_to_goal())
        last_metrics['action_distance_to_goal'].append(
                len(final_expert_actions))
        last_metrics['trajectory_length'].append(
                len(trajectory_results['frames']))
        # Record average policy entropy over an episode
        with torch.no_grad():
            entropy = trajectory_avg_entropy(all_action_scores)
        last_metrics['entropy'].append(entropy.item())

        for metric in last_metrics.keys():
            writer.add_scalar('train/' + metric, last_metrics[metric][-1],
                    train_iter)
            writer.add_scalar('train/frames_' + metric, last_metrics[metric][-1],
                    train_frames)

        # Evaluate and save checkpoint every N trajectories, collect/print stats
        if train_iter % eval_interval == 0:
            print('iteration %d frames %d' % (train_iter, train_frames))
            for metric, values in last_metrics.items():
                mean = np.mean(values)
                writer.add_scalar('train_avg/' + metric, mean, train_iter)
                writer.add_scalar('train_avg/frames_' + metric, mean,
                        train_frames)
                last_metrics[metric] = []

            # Collect validation statistics and write, print
            results = eval_online(fo, model, frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    seen_episodes=eval_episodes_seen,
                    unseen_episodes=eval_episodes_unseen)

            write_eval_results(writer, results, train_iter, train_frames)

            if save_path is not None:
                if save_intermediate:
                    checkpoint_save_path = os.path.join(save_path, 'model_' +
                            str(train_iter) + '.pth')
                else:
                    checkpoint_save_path = os.path.join(save_path, 'model.pth')
                print('saving to ' + checkpoint_save_path)
                torch.save({
                    'train_iter' : train_iter,
                    'train_frames' : train_frames,
                    # Save train_trajectories to be compatible with train_dataset
                    'train_trajectories' : train_iter,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, checkpoint_save_path)

def write_eval_results(writer, results, train_iter, train_frames,
        train_trajectories=None):
    for split in results.keys():
        for metric, values in results[split].items():
            mean = np.mean(values)
            writer.add_scalar('validation_online_avg/' + split + '/' + metric,
                    mean, train_iter)
            writer.add_scalar('validation_online_avg/' + split + '/frames' +
                    metric, mean, train_frames)
            if train_trajectories is not None:
                writer.add_scalar('validation_online_avg/' + split +
                        '/trajectories' + metric, mean, train_trajectories)

def eval_online(fo, model, frame_stack=1, zero_fill_frame_stack=False,
        seen_episodes=1, unseen_episodes=1):
    model.eval()
    metrics = {}
    for split in ['seen', 'unseen']:
        metrics[split] = {}
        metrics[split]['success'] = []
        metrics[split]['path_weighted_success'] = []
        metrics[split]['crow_distance_to_goal'] = []
        metrics[split]['walking_distance_to_goal'] = []
        metrics[split]['action_distance_to_goal'] = []
        metrics[split]['entropy'] = []
        metrics[split]['trajectory_length'] = []
        episodes = seen_episodes if split == 'seen' else unseen_episodes
        for i in range(episodes):
            with torch.no_grad():
                trajectory_results = rollout_trajectory(fo, model,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        scene_name_or_num=random.choice(TRAIN_SCENE_NUMBERS if
                            split == 'seen' else VALID_UNSEEN_SCENE_NUMBERS))
            metrics[split]['success'].append(
                    float(trajectory_results['success']))
            metrics[split]['path_weighted_success'].append(
                    path_weighted_success(
                        trajectory_results['success'],
                        len(trajectory_results['frames']),
                        len(fo.original_expert_actions)))
            final_expert_actions, final_expert_path = \
                    fo.get_current_expert_actions_path()
            metrics[split]['crow_distance_to_goal'].append(
                    fo.crow_distance_to_goal())
            metrics[split]['walking_distance_to_goal'].append(
                    fo.walking_distance_to_goal())
            metrics[split]['action_distance_to_goal'].append(
                    len(final_expert_actions))
            with torch.no_grad():
                entropy = trajectory_avg_entropy(torch.cat(
                    trajectory_results['all_action_scores']))
            metrics[split]['entropy'].append(entropy.item())
            metrics[split]['trajectory_length'].append(
                    float(len(trajectory_results['frames'])))

    model.train()

    return metrics

if __name__ == '__main__':
    args = parser.parse_args()

    if args.load_path is not None and not os.path.isfile(args.load_path):
        print('load_path not found: ' + args.load_path)
        exit()

    if args.save_path is not None and not os.path.isdir(args.save_path):
        print('making save_path: ' + args.save_path)
        os.makedirs(args.save_path)

    env = ThorEnv()

    if args.dataset_path is None:
        obj_type_to_index_path = os.path.join(os.environ['ALFRED_ROOT'], 'env',
                'obj_type_to_index.json')
        if not os.path.isfile(obj_type_to_index_path):
            obj_type_to_index = index_all_items(env)
            with open(obj_type_to_index_path, 'w') as jsonfile:
                json.dump(obj_type_to_index, jsonfile)
    else:
        obj_type_to_index_path = os.path.join(args.dataset_path,
                'obj_type_to_index.json')

    with open(obj_type_to_index_path, 'r') as jsonfile:
        obj_type_to_index = json.load(jsonfile)
    print(obj_type_to_index)
    index_to_obj_type = {i: ot for ot, i in obj_type_to_index.items()}

    fo = FindOne(env, obj_type_to_index)

    if args.dataset_path is not None:
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

    if args.dataset_path is not None:
        train_dataset(fo, model, optimizer, dataloaders, obj_type_to_index,
                dataset_transitions=args.dataset_transitions,
                batch_size=args.batch_size, frame_stack=args.frame_stack,
                zero_fill_frame_stack=args.zero_fill_frame_stack,
                eval_episodes_seen=args.eval_episodes_seen,
                eval_episodes_unseen=args.eval_episodes_unseen,
                eval_interval=args.eval_interval, save_path=args.save_path,
                save_intermediate=args.save_intermediate,
                load_path=args.load_path)
    else:
        train(fo, model, optimizer, frame_stack=args.frame_stack,
                zero_fill_frame_stack=args.zero_fill_frame_stack,
                teacher_force=args.teacher_force,
                eval_episodes_seen=args.eval_episodes_seen,
                eval_episodes_unseen=args.eval_episodes_unseen,
                eval_interval=args.eval_interval, save_path=args.save_path,
                save_intermediate=args.save_intermediate,
                load_path=args.load_path)

