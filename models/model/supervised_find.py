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
from models.nn.resnet import Resnet
from models.nn.fo import LateFusion, NatureCNN, FCPolicy, LSTMPolicy, \
        ObjectEmbedding
import gen.constants as constants
from gen.graph.graph_obj import Graph
from data.fo_dataset import get_datasets_dataloaders
from models.utils.metric import compute_actions_f1
from utils.video_util import VideoSaver

video_saver = VideoSaver()

from tensorboardX import SummaryWriter

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-lr', '--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-oed', '--object-embedding-dim', type=int, default=16, help='object embedding dim')
parser.add_argument('-fs', '--frame-stack', type=int, default=1, help='number of frames to stack')
parser.add_argument('-zffs', '--zero-fill-frame-stack', dest='zero_fill_frame_stack', action='store_true', help='fill frames with zeros when frame stacking on early steps')
parser.add_argument('-fffs', '--first-fill-frame-stack', dest='zero_fill_frame_stack', action='store_false', help='replicate first frame when frame stacking on early steps')
parser.set_defaults(zero_fill_frame_stack=False)
parser.add_argument('-vm', '--visual-model', type=str, default='naturecnn', help='which visual model to use (\'naturecnn\', \'resnet\', or \'maskrcnn\')')
parser.add_argument('-pm', '--policy-model', type=str, default='fc', help='which policy model to use (\'fc\' or \'lstm\')')
parser.add_argument('-fl', '--fc-layers', type=int, default=1, help='number of fc layers')
parser.add_argument('-lhs', '--lstm-hidden-dim', type=int, default=64, help='hidden dimension of lstm layers (only used if not using object embedding as initial hidden state)')
parser.add_argument('-ll', '--lstm-layers', type=int, default=1, help='number of lstm layers')
parser.add_argument('-ilo', '--init-lstm-object', dest='init_lstm_object', action='store_true', help='use object embedding to initialize lstm hidden state')
parser.add_argument('-ilz', '--init-lstm-zero', dest='init_lstm_object', action='store_false', help='initialize lstm hidden state to zeros')
parser.set_defaults(init_lstm_object=False)
parser.add_argument('-rcf', '--resnet-conv-feat', dest='resnet_conv_feat', action='store_true', help='use conv feat (skip last two resnet-18 layers)')
parser.add_argument('-rll', '--resnet-last-layers', dest='resnet_conv_feat', action='store_false', help='use last two resnet-18 layers')
parser.set_defaults(resnet_conv_feat=True)
parser.add_argument('-ei', '--eval-interval', type=int, default=100, help='number of training steps between evaluations')
parser.add_argument('-te', '--train-episodes', type=int, default=10, help='number of episodes to evaluate live on train seen scenes and trajectories')
parser.add_argument('-vse', '--valid-seen-episodes', type=int, default=10, help='number of episodes to evaluate live on validation seen scenes and trajectories')
parser.add_argument('-vue', '--valid-unseen-episodes', type=int, default=10, help='number of episodes to evaluate live on validation unseen scenes and trajectories')
parser.add_argument('-tf', '--teacher-force', dest='teacher_force', action='store_true')
parser.add_argument('-ntf', '--no-teacher-force', dest='teacher_force', action='store_false')
parser.set_defaults(teacher_force=False)
parser.add_argument('-dp', '--dataset-path', type=str, default=None, help='path (directory) to dataset indexes of trajectories and obj_type_to_index, if using')
parser.add_argument('-bs', '--batch-size', type=int, default=1, help='batch size of training trajectories')
parser.add_argument('-ms', '--max-steps', type=float, default=100000, help='max training gradient steps')
parser.add_argument('-do', '--dropout', type=float, default=0.0, help='dropout prob')
parser.add_argument('-sp', '--save-path', type=str, default=None, help='path (directory) to save models and tensorboard stats')
parser.add_argument('-si', '--save-intermediate', dest='save_intermediate', action='store_true', help='save intermediate checkpoints (once per eval interval)')
parser.add_argument('-nsi', '--no-save-intermediate', dest='save_intermediate', action='store_false', help='don\'t save intermediate checkpoints (once per eval interval)')
parser.set_defaults(save_intermediate=False)
parser.add_argument('-sv', '--save-images-video', dest='save_images_video', action='store_true', help='save images and video (for trajectories, every eval interval)')
parser.add_argument('-nsv', '--no-save-images-video', dest='save_images_video', action='store_false', help='don\'t save images and video (for trajectories, every eval interval)')
parser.set_defaults(save_images_video=False)
parser.add_argument('-lp', '--load-path', type=str, default=None, help='path (.pth) to load model checkpoint from')
parser.add_argument('-g', '--gpu', type=int, default=3, help='GPU to run model on')
'''
parser.add_argument('-do', '--dropout', type=float, default=0.02, help='dropout prob')
parser.add_argument('-sn', '--save-name', type=str, default='model', help='model save name')
parser.add_argument('-id', '--model-id', type=str, default='model', help='model id')
'''

args = parser.parse_args()

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
device = torch.device('cuda:' + str(args.gpu))

def trajectory_avg_entropy(trajectory_logits):
    return -torch.mean(torch.sum(
            F.log_softmax(trajectory_logits, dim=-1) *
            torch.exp(F.log_softmax(trajectory_logits, dim=-1)),
            dim=-1), dim=-1)

def path_weighted_success(success, num_agent_actions, num_expert_actions):
    return float(success) * num_expert_actions / max(num_agent_actions,
            num_expert_actions)

def rollout_trajectory(fo, model, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, scene_name_or_num=None, traj_data=None,
        high_idx=None):
    """
    Returns dictionary of trajectory results, or None if failed to load from
    traj_data dict with given high_idx.
    """
    frames = []
    all_action_scores = []
    pred_action_indexes = []
    expert_action_indexes = []
    if traj_data is not None:
        frames, target_object_type_index = fo.load_from_traj_data(traj_data,
                high_idx=high_idx)
        if frames is None: # Failed to load from traj data
            return None
    else:
        frame, target_object_type_index = fo.reset(scene_name_or_num)
        frames.append(frame)
    done = False
    initial_crow_distance = fo.crow_distance_to_goal()
    model.reset_hidden(batch_size=1, device=device)
    while not done:
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

        action_scores = model.predict([stacked_frames],
                torch.tensor([target_object_type_index], device=device),
                use_hidden=True)
        # Sorted in increasing order (rightmost is highest scoring action)
        sorted_scores, top_indices = torch.sort(action_scores[0],
                descending=True)
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
        frames.append(frame)
        all_action_scores.append(action_scores[0])
        pred_action_indexes.append(pred_action_index)
        expert_action_indexes.append(ACTION_TO_INDEX[expert_action])

    print('trajectory len: ' + str(len(all_action_scores)))

    success = fo.get_success()
    trajectory_results = {}
    trajectory_results['scene_name_or_num'] = fo.get_scene_name_or_num()
    trajectory_results['target'] = fo.get_target_object_type()
    trajectory_results['initial_action_distance'] = \
            len(fo.original_expert_actions)
    trajectory_results['initial_crow_distance'] = initial_crow_distance
    accuracy, f1 = actions_accuracy_f1(torch.Tensor(pred_action_indexes),
            torch.Tensor(expert_action_indexes))
    trajectory_results['accuracy'] = accuracy
    trajectory_results['f1'] = f1
    trajectory_results['frames'] = frames
    trajectory_results['all_action_scores'] = all_action_scores
    trajectory_results['pred_action_indexes'] = pred_action_indexes
    trajectory_results['expert_action_indexes'] = expert_action_indexes
    trajectory_results['success'] = float(success)
    trajectory_results['path_weighted_success'] = path_weighted_success(
            success, len(frames), len(fo.original_expert_actions))
    trajectory_results['crow_distance_to_goal'] = fo.crow_distance_to_goal()
    trajectory_results['walking_distance_to_goal'] = \
            fo.walking_distance_to_goal()
    final_expert_actions, final_expert_path = \
            fo.get_current_expert_actions_path()
    trajectory_results['action_distance_to_goal'] = len(final_expert_actions)
    trajectory_results['target_visible'] = fo.target_visible()
    trajectory_results['expert_actions'] = fo.get_original_expert_actions()
    with torch.no_grad():
        entropy = trajectory_avg_entropy(torch.cat(all_action_scores))
    trajectory_results['entropy'] = entropy.item()
    return trajectory_results

def stack_frames(frames, frame_stack=1, zero_fill_frame_stack=False):
    stacked_frames = []
    for trajectory_index in range(len(frames)):
        trajectory_frames = []
        for transition_index in range(len(frames[trajectory_index])):
            if transition_index < frame_stack - 1:
                if zero_fill_frame_stack:
                    # Fill earlier frames with zeroes
                    transition_frames = torch.cat([torch.zeros(((frame_stack -
                        transition_index - 1) * 3), 300, 300)] +
                        [frame.permute(2, 0, 1) for frame in
                            frames[trajectory_index][:transition_index+1]])
                else:
                    # Repeat first frame
                    transition_frames = torch.cat([frames[trajectory_index][0]
                        .permute(2, 0, 1) for _ in range(frame_stack -
                            transition_index - 1)] +
                        [frame.permute(2, 0, 1) for frame in
                            frames[trajectory_index][:transition_index+1]])
                trajectory_frames.append(transition_frames)
            else:
                trajectory_frames.append(torch.cat([frame.permute(2, 0, 1) for
                    frame in frames[trajectory_index][
                        transition_index-frame_stack+1:transition_index+1]]))
        stacked_frames.append(torch.stack(trajectory_frames).to(device=device,
            dtype=torch.float32))
    return stacked_frames

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
    f1 = compute_actions_f1(
            predicted_action_indexes,
            expert_action_indexes).item()
    return accuracy, f1

def write_images_video(results_online, train_steps, save_path):
    for split in results_online.keys():
        for trajectory_index in range(len(results_online[split]['frames'])):
            trajectory_images = results_online[split] \
                    ['frames'][trajectory_index]
            target = results_online[split]['target'][trajectory_index]
            scene_name_or_num = str(results_online[split]['scene_name_or_num']
                    [trajectory_index])
            success_or_failure = 'success' if results_online[split] \
                    ['success'][trajectory_index] else 'failure'
            crow_distance = str(results_online[split]['crow_distance_to_goal']
                    [trajectory_index])
            action_distance = str(results_online[split]
                    ['action_distance_to_goal'][trajectory_index])
            target_visible = 'target_visible' if results_online[split] \
                    ['target_visible'][trajectory_index] else \
                    'target_not_visible'
            trajectory_length = str(results_online[split]['trajectory_length']
                    [trajectory_index])
            images_save_path =  os.path.join(save_path, 'images_video',
                    str(train_steps), split, str(trajectory_index) + '_' +
                    target + '_' + 'scene' + scene_name_or_num + '_' +
                    success_or_failure + '_' + 'crowdist' + crow_distance + '_'
                    + 'actiondist' + action_distance + '_' + target_visible +
                    '_' + 'trajectorylen' + trajectory_length)
            if not os.path.isdir(images_save_path):
                os.makedirs(images_save_path)

            for image_index in range(len(trajectory_images)):
                image_save_path = os.path.join(images_save_path, '%09d.png' %
                        int(image_index))
                cv2.imwrite(image_save_path, trajectory_images[image_index])
            video_save_path = os.path.join(images_save_path,
                    'video.mp4')
            video_saver.save(os.path.join(images_save_path,
                '*.png'), video_save_path)

def train_dataset(fo, model, optimizer, datasets, dataloaders,
        obj_type_to_index, batch_size=1, frame_stack=1,
        zero_fill_frame_stack=False, train_episodes=10, valid_seen_episodes=10,
        valid_unseen_episodes=10, eval_interval=100, max_steps=100000,
        save_path=None, save_intermediate=False, save_images_video=False,
        load_path=None):
    """
    Train a model by sampling from a torch dataloader.

    dataloaders is a dict with 'train', 'valid_seen', 'valid_unseen' as keys
    and values of those dataloaders.
    """
    writer = SummaryWriter(log_dir='tensorboard_logs' if save_path is None
            else os.path.join(save_path, 'tensorboard_logs'))
    if load_path is not None:
        checkpoint = torch.load(load_path)
        train_steps = checkpoint['train_steps']
        train_frames = checkpoint['train_frames']
        train_trajectories = checkpoint['train_trajectories']
        train_epochs = checkpoint['train_epochs']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading from ' + load_path + ' epoch ' + str(train_epochs))
    else:
        train_steps = 0
        train_frames = 0
        train_trajectories = 0
        train_epochs = 0
    last_metrics = {}
    last_metrics['loss'] = []
    last_metrics['accuracy'] = []
    last_metrics['f1'] = []
    last_metrics['entropy'] = []

    while train_steps < max_steps:
        for batch_samples in dataloaders['train']:
            # These are all lists of tensors where each tensor corresponds to a
            # variable-length trajectory
            stacked_frames = stack_frames(batch_samples['images'],
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack)
            # Turn target names into indexes and action names into indexes
            target_indexes = torch.tensor([obj_type_to_index[
                constants.OBJECTS_LOWER_TO_UPPER[target]] for target in
                batch_samples['target']], device=device)
            action_indexes = [
                    torch.tensor([ACTION_TO_INDEX[action] for action in
                        trajectory_actions], device=device)
                    for trajectory_actions in batch_samples['low_actions']]

            # Train
            action_scores = model(stacked_frames, target_indexes)
            loss = F.cross_entropy(torch.cat(action_scores),
                    torch.cat(action_indexes))
            optimizer.zero_grad()
            # TODO: RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
            #loss.backward()
            try:
                loss.backward(retain_graph=True)
            except:
                loss.backward()
            # TODO: may want to clamp gradients
            optimizer.step()

            train_steps += 1
            train_frames += sum([scores.shape[0] for scores in action_scores])
            train_trajectories += batch_size
            last_metrics['loss'].append(loss.item())
            accuracy, f1 = actions_accuracy_f1(torch.argmax(
                torch.cat(action_scores), dim=1), torch.cat(action_indexes))
            last_metrics['accuracy'].append(accuracy)
            last_metrics['f1'].append(f1)
            # Record average policy entropy over all trajectories/transitions
            with torch.no_grad():
                entropy = trajectory_avg_entropy(torch.cat(action_scores))
            last_metrics['entropy'].append(entropy.item())

            results = {}
            results['train'] = {}
            for metric in last_metrics.keys():
                results['train'][metric] = last_metrics[metric][-1]
            write_results(writer, results, train_steps, train_frames,
                    train_trajectories=train_trajectories, method='dataset',
                    save_path=None) # Don't write training results to file

            # Also run an evaluation if this is the last gradient step
            if train_steps % eval_interval == 0 or train_steps == max_steps:
                print('epoch %d steps %d frames %d trajectories %d' %
                        (train_epochs, train_steps, train_frames,
                            train_trajectories))
                results = {}
                results['train'] = {}
                for metric, values in last_metrics.items():
                    results['train']['avg/' + metric] = values
                    print('avg ' + metric + ' %.6f' % np.mean(values))
                    last_metrics[metric] = []
                write_results(writer, results, train_steps, train_frames,
                        train_trajectories=train_trajectories,
                        train_epochs=train_epochs, method='dataset',
                        save_path=None)

                # Evaluate on valid_seen and valid_unseen
                results_dataset = eval_dataset(model, dataloaders,
                        obj_type_to_index,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack)
                write_results(writer, results_dataset, train_steps, train_frames,
                        train_trajectories=train_trajectories,
                        train_epochs=train_epochs, method='dataset',
                        save_path=save_path)

                for split in ['valid_seen', 'valid_unseen']:
                    print(split + ' accuracy %.6f f1 %.6f' %
                            (results_dataset[split]['accuracy'],
                                results_dataset[split]['f1']))

                # Evaluate online in environment but load training trajectories
                # from datasets
                results_trajectory = eval_online(fo, model,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        train_episodes=train_episodes,
                        valid_seen_episodes=valid_seen_episodes,
                        valid_unseen_episodes=valid_unseen_episodes,
                        datasets=datasets)
                write_results(writer, results_trajectory, train_steps,
                        train_frames, train_trajectories, train_epochs,
                        method='trajectory', save_path=save_path)

                results_online = eval_online(fo, model,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        train_episodes=train_episodes,
                        valid_seen_episodes=valid_seen_episodes,
                        valid_unseen_episodes=valid_unseen_episodes)
                write_results(writer, results_online, train_steps, train_frames,
                        train_trajectories, train_epochs, method='online',
                        save_path=save_path)

                if save_path is not None:
                    if save_intermediate:
                        checkpoint_save_path = os.path.join(save_path, 'model_' +
                                str(train_steps) + '.pth')
                    else:
                        checkpoint_save_path = os.path.join(save_path, 'model.pth')
                    print('saving to ' + checkpoint_save_path)
                    torch.save({
                        'train_steps' : train_steps,
                        'train_frames' : train_frames,
                        'train_trajectories' : train_trajectories,
                        'train_epochs' : train_epochs,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                    }, checkpoint_save_path)

                    if save_images_video:
                        write_images_video(results_online, train_steps, save_path)

        train_epochs += 1

def eval_dataset(model, dataloaders, obj_type_to_index, frame_stack=1,
        zero_fill_frame_stack=False):
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
            stacked_frames = stack_frames(batch_samples['images'],
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack)

            # Turn target names into indexes and action names into indexes
            target_indexes = torch.tensor([obj_type_to_index[
                constants.OBJECTS_LOWER_TO_UPPER[target]] for target in
                batch_samples['target']], device=device)
            action_indexes = [
                    torch.tensor([ACTION_TO_INDEX[action] for action in
                        trajectory_actions], device=device)
                    for trajectory_actions in batch_samples['low_actions']]

            with torch.no_grad():
                action_scores = model(stacked_frames, target_indexes)
            all_predicted_action_indexes.append(torch.argmax(
                torch.cat(action_scores), dim=1))
            all_expert_action_indexes.append(torch.cat(action_indexes))

        all_predicted_action_indexes = torch.cat(all_predicted_action_indexes)
        all_expert_action_indexes = torch.cat(all_expert_action_indexes)

        accuracy, f1 = actions_accuracy_f1(all_predicted_action_indexes,
                all_expert_action_indexes)
        results[split]['accuracy'] = accuracy
        results[split]['f1'] = f1

    model.train()
    return results


def train_online(fo, model, optimizer, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, train_episodes=10, valid_seen_episodes=10,
        valid_unseen_episodes=10, eval_interval=1000, max_steps=100000,
        save_path=None, save_intermediate=False, save_images_video=False,
        load_path=None):
    """
    Train a model by collecting a trajectory online, then training with correct
    action supervision. Loads model from checkpoint if load_path is not None.
    """
    writer = SummaryWriter(log_dir='tensorboard_logs' if save_path is None
            else os.path.join(save_path, 'tensorboard_logs'))
    if load_path is not None:
        checkpoint = torch.load(load_path)
        train_steps = checkpoint['train_steps']
        train_frames = checkpoint['train_frames']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading from ' + load_path + ' iteration ' + str(train_steps))
    else:
        train_steps = 0
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
    last_metrics['target_visible'] = []
    last_metrics['trajectory_length'] = []
    last_metrics['entropy'] = []

    # TODO: want a replay memory?
    while train_steps < max_steps:
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
        train_steps += 1
        train_frames += len(trajectory_results['frames'])
        last_metrics['loss'].append(loss.item())
        last_metrics['success'].append(float(trajectory_results['success']))
        last_metrics['path_weighted_success'].append(
                path_weighted_success(
                    trajectory_results['success'],
                    len(trajectory_results['frames']),
                    len(fo.get_original_expert_actions())))

        final_expert_actions, final_expert_path = \
                fo.get_current_expert_actions_path()
        last_metrics['crow_distance_to_goal'].append(
                fo.crow_distance_to_goal())
        last_metrics['walking_distance_to_goal'].append(
                fo.walking_distance_to_goal())
        last_metrics['action_distance_to_goal'].append(
                len(final_expert_actions))
        last_metrics['target_visible'].append(fo.target_visible())
        last_metrics['trajectory_length'].append(
                len(trajectory_results['frames']))
        # Record average policy entropy over an episode
        with torch.no_grad():
            entropy = trajectory_avg_entropy(all_action_scores)
        last_metrics['entropy'].append(entropy.item())

        results = {}
        results['train'] = {}
        for metric in last_metrics.keys():
            results['train'][metric] = last_metrics[metric][-1]
        # Don't write training results to file
        write_results(writer, results, train_steps, train_frames,
                method='online', save_path=None)

        # Evaluate and save checkpoint every N trajectories, collect/print
        # stats
        if train_steps % eval_interval == 0 or train_steps == max_steps:
            print('steps %d frames %d' % (train_steps, train_frames))
            results = {}
            results['train'] = {}
            for metric, values in last_metrics.items():
                results['train']['avg/' + metric] = values
                last_metrics[metric] = []
            write_results(writer, results, train_steps, train_frames,
                    method='online', save_path=None)

            # Collect validation statistics and write, print
            results = eval_online(fo, model, frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    train_episodes=train_episodes,
                    valid_seen_episodes=valid_seen_episodes,
                    valid_unseen_episodes=valid_unseen_episodes)

            write_results(writer, results, train_steps, train_frames,
                    method='online', save_path=save_path)

            if save_path is not None:
                if save_intermediate:
                    checkpoint_save_path = os.path.join(save_path, 'model_' +
                            str(train_steps) + '.pth')
                else:
                    checkpoint_save_path = os.path.join(save_path, 'model.pth')
                print('saving to ' + checkpoint_save_path)
                torch.save({
                    'train_steps' : train_steps,
                    'train_frames' : train_frames,
                    # Save train_trajectories and train_epochs to be compatible
                    # with train_dataset
                    'train_trajectories' : train_steps,
                    'train_epochs' : train_steps,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, checkpoint_save_path)

                if save_images_video:
                    write_images_video(results, train_steps, save_path)

def write_results(writer, results, train_steps, train_frames,
        train_trajectories=None, train_epochs=None, method='dataset',
        save_path=None):
    """
    Write results to SummaryWriter. Method is either "dataset" or "online"
    depending on whether the metrics were acquired in a traditional
    supervised-learning method or are from unrolling (with expert trajectories)
    in the environment.
    """
    for split in results.keys():
        for metric, values in results[split].items():
            if metric in ['frames', 'target', 'scene_name_or_num',
                    'trajectory_index', 'initial_action_distance',
                    'initial_crow_distance']:
                continue
            mean = np.mean(values)
            writer.add_scalar('steps/' + split + '/' + method + '/' + metric,
                    mean, train_steps)
            writer.add_scalar('frames/' + split + '/' + method + '/' + metric,
                    mean, train_frames)
            if train_trajectories is not None:
                writer.add_scalar('trajectories/' + split + '/' + method + '/'
                        + metric, mean, train_trajectories)
            if train_epochs is not None:
                writer.add_scalar('epochs/' + split + '/' + method + '/' +
                        metric, mean, train_epochs)
    # Also write output to saved file
    if save_path is not None:
        results_path = os.path.join(save_path, str(train_steps))
        # Exclude frames from results
        json_results = {}
        for split in results.keys():
            json_results[split] = {k:v for k, v in results[split].items() if k
                    != 'frames'}
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        with open(os.path.join(results_path, method + '.json'), 'w') as \
                jsonfile:
            json.dump(json_results, jsonfile)

def eval_online(fo, model, frame_stack=1, zero_fill_frame_stack=False,
        train_episodes=1, valid_seen_episodes=1, valid_unseen_episodes=1,
        datasets=None):
    """
    Evaluate by gathering live rollouts in the environment. If dataloaders is
    not None, will sample trajectories from the provided dict of datasets
    ('train', 'valid_seen', and 'valid_unseen') and set the scene to those.
    """
    model.eval()
    metrics = {}
    for split, episodes, scene_numbers in zip(
            ['train', 'valid_seen', 'valid_unseen'],
            [train_episodes, valid_seen_episodes, valid_unseen_episodes],
            [TRAIN_SCENE_NUMBERS, VALID_SEEN_SCENE_NUMBERS,
                VALID_UNSEEN_SCENE_NUMBERS]):
        metrics[split] = {}
        metrics[split]['target'] = []
        metrics[split]['initial_action_distance'] = []
        metrics[split]['initial_crow_distance'] = []
        metrics[split]['accuracy'] = []
        metrics[split]['f1'] = []
        metrics[split]['success'] = []
        metrics[split]['path_weighted_success'] = []
        metrics[split]['crow_distance_to_goal'] = []
        metrics[split]['walking_distance_to_goal'] = []
        metrics[split]['action_distance_to_goal'] = []
        metrics[split]['target_visible'] = []
        metrics[split]['entropy'] = []
        metrics[split]['trajectory_length'] = []
        metrics[split]['frames'] = []
        metrics[split]['scene_name_or_num'] = []
        if datasets is not None:
            metrics[split]['trajectory_index'] = []
        for i in range(episodes):
            with torch.no_grad():
                if datasets is not None:
                    # Sample a trajectory
                    load_success = False
                    while not load_success:
                        trajectory_index = random.randint(0,
                                len(datasets[split]))
                        trajectory = datasets[split].trajectories[
                                trajectory_index]
                        with open(os.path.join(trajectory['path'],
                            'traj_data.json'), 'r') as jsonfile:
                            traj_data = json.load(jsonfile)
                        trajectory_results = rollout_trajectory(fo, model,
                                frame_stack=frame_stack,
                                zero_fill_frame_stack=zero_fill_frame_stack,
                                traj_data=traj_data,
                                high_idx=trajectory['high_idx'][0])
                        load_success = trajectory_results is not None
                else:
                    trajectory_results = rollout_trajectory(fo, model,
                            frame_stack=frame_stack,
                            zero_fill_frame_stack=zero_fill_frame_stack,
                            scene_name_or_num=random.choice(scene_numbers))
            # TODO: append in a loop over keys here and elsewhere
            metrics[split]['target'].append(trajectory_results['target'])
            metrics[split]['initial_action_distance'].append(
                    trajectory_results['initial_action_distance'])
            metrics[split]['initial_crow_distance'].append(
                    trajectory_results['initial_crow_distance'])
            metrics[split]['accuracy'].append(trajectory_results['accuracy'])
            metrics[split]['f1'].append(trajectory_results['f1'])
            metrics[split]['success'].append(trajectory_results['success'])
            metrics[split]['path_weighted_success'].append(
                    trajectory_results['path_weighted_success'])
            metrics[split]['crow_distance_to_goal'].append(
                    trajectory_results['crow_distance_to_goal'])
            metrics[split]['walking_distance_to_goal'].append(
                    trajectory_results['walking_distance_to_goal'])
            metrics[split]['action_distance_to_goal'].append(
                    trajectory_results['action_distance_to_goal'])
            metrics[split]['target_visible'].append(
                    trajectory_results['target_visible'])
            metrics[split]['entropy'].append(trajectory_results['entropy'])
            metrics[split]['trajectory_length'].append(
                    float(len(trajectory_results['frames'])))
            metrics[split]['frames'].append(trajectory_results['frames'])
            metrics[split]['scene_name_or_num'].append(
                    trajectory_results['scene_name_or_num'])
            if datasets is not None:
                metrics[split]['trajectory_index'].append(trajectory_index)

    model.train()

    return metrics

if __name__ == '__main__':
    if args.load_path is not None and not os.path.isfile(args.load_path):
        print('load_path not found: ' + args.load_path)
        exit()

    if args.save_path is not None and not os.path.isdir(args.save_path):
        print('making save_path: ' + args.save_path)
        os.makedirs(args.save_path)
        images_video_save_path = os.path.join(args.save_path, 'images_video')
        if args.save_images_video and not os.path.isdir(images_video_save_path):
            os.makedirs(images_video_save_path)

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
        datasets, dataloaders = get_datasets_dataloaders(
                batch_size=args.batch_size)
    if args.visual_model.lower() == 'nature' or args.visual_model.lower() == \
            'naturecnn':
        visual_model = NatureCNN(frame_stack=args.frame_stack)
        visual_output_size = visual_model.output_size
    elif args.visual_model.lower() == 'resnet' or args.visual_model.lower() \
            == 'resnet18' or args.visual_model.lower() == 'maskrcnn':
        # Resnet class is object, not nn.Module
        args.gpu = True
        visual_model = Resnet(args, share_memory=True,
                use_conv_feat=args.resnet_conv_feat)
        # Visual features are (frame)stacked, not frames, with pretrained
        # models. LateFusion class takes care of that
        visual_output_size = visual_model.output_size * args.frame_stack
    object_embeddings = ObjectEmbedding(len(obj_type_to_index),
            args.object_embedding_dim)
    if args.policy_model.lower() == 'fc':
        policy_model = FCPolicy(visual_output_size + args.object_embedding_dim,
                NUM_ACTIONS, num_fc_layers=args.fc_layers)
    elif args.policy_model.lower() == 'lstm':
        policy_model = LSTMPolicy(visual_output_size +
                args.object_embedding_dim, NUM_ACTIONS,
                lstm_hidden_dim=args.object_embedding_dim if
                args.init_lstm_object else args.lstm_hidden_dim,
                num_lstm_layers=args.lstm_layers, dropout=args.dropout,
                num_fc_layers=args.fc_layers,
                init_lstm_object=args.init_lstm_object)

    try:
        model = LateFusion(visual_model, object_embeddings,
                policy_model).to(device)
    except:
        model = LateFusion(visual_model, object_embeddings,
                policy_model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.dataset_path is not None:
        train_dataset(fo, model, optimizer, datasets, dataloaders,
                obj_type_to_index,
                batch_size=args.batch_size, frame_stack=args.frame_stack,
                zero_fill_frame_stack=args.zero_fill_frame_stack,
                train_episodes=args.train_episodes,
                valid_seen_episodes=args.valid_seen_episodes,
                valid_unseen_episodes=args.valid_unseen_episodes,
                eval_interval=args.eval_interval, max_steps=args.max_steps,
                save_path=args.save_path,
                save_intermediate=args.save_intermediate,
                save_images_video=args.save_images_video,
                load_path=args.load_path)
    else:
        train_online(fo, model, optimizer, frame_stack=args.frame_stack,
                zero_fill_frame_stack=args.zero_fill_frame_stack,
                teacher_force=args.teacher_force,
                train_episodes=args.train_episodes,
                valid_seen_episodes=args.valid_seen_episodes,
                valid_unseen_episodes=args.valid_unseen_episodes,
                eval_interval=args.eval_interval, max_steps=args.max_steps,
                save_path=args.save_path,
                save_intermediate=args.save_intermediate,
                save_images_video=args.save_images_video,
                load_path=args.load_path)

