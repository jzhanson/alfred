import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import random
import copy

import numpy as np
import torch
import torch.nn.functional as F

import gen.constants as constants
from env.interaction_exploration import InteractionExploration
from models.utils.metric import per_step_entropy, trajectory_avg_entropy
from models.utils.helper_utils import stack_frames
from utils.video_util import VideoSaver

video_saver = VideoSaver()

from tensorboardX import SummaryWriter

from args import parse_args

"""
There's a "bug" where the underlying Unity environment doesn't like being
passed objects for ToggleObjectOn/Off that aren't visible and throws an
exception that's caught by env/thor_env.py, but seems to work fine for other
interactions with not visible objects. This is a small issue since the
"visible" distance is not very large in the THOR environment.
"""

def rollout_trajectory(env, model, single_interact=False, use_masks=True,
        fusion_model='SuperpixelFusion', max_trajectory_length=None,
        frame_stack=1, zero_fill_frame_stack=False, teacher_force=False,
        sample_action=True, sample_mask=True, scene_name_or_num=None,
        reset_kwargs={}, device=torch.device('cpu')):
    """
    Returns dictionary of trajectory results.
    """
    frames = []
    action_successes = []
    all_action_scores = []
    values = []
    pred_action_indexes = []
    rewards = []
    expert_action_indexes = []
    if fusion_model == 'SuperpixelFusion':
        all_mask_scores = []
    frame = env.reset(scene_name_or_num, **reset_kwargs)
    done = False
    num_steps = 0

    actions = (constants.SIMPLE_ACTIONS if single_interact else
            constants.COMPLEX_ACTIONS)
    action_to_index = (constants.ACTION_TO_INDEX_SIMPLE if single_interact else
            constants.ACTION_TO_INDEX_COMPLEX)

    if fusion_model == 'SuperpixelFusion':
        prev_action_index = None
    elif fusion_model == 'SuperpixelActionConcat':
        prev_action_features = None
    hidden_state = model.init_policy_hidden(batch_size=1, device=device)
    while not done and (max_trajectory_length is None or num_steps <
            max_trajectory_length):
        frames.append(torch.from_numpy(np.ascontiguousarray(frame)))
        current_expert_actions, _ = env.get_current_expert_actions_path()

        # stack_frames takes a list of tensors, one tensor per trajectory, so
        # wrap frames in an outer list and unwrap afterwards. Also,
        # stack_frames needs the previous frame_stack frames, so pass the
        # required number of frames but only take the last stacked frame of
        # that list
        # Put these frames on cpu because pre-Resnet transforms require not
        # CUDA tensors
        stacked_frames = stack_frames([frames[-frame_stack:]],
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
                device=torch.device('cpu'))[0][-1:]

        if fusion_model == 'SuperpixelFusion':
            if prev_action_index is not None:
                prev_action_index = [torch.LongTensor([prev_action_index])
                        .to(device)]
            else:
                # Not super worried about null/no prev action not having a separate
                # embedding - all zeros is fine since it's the input to the LSTM
                # TODO: add separate embedding for null/no previous action
                # Let SuperpixelFusion's forward take care of this
                prev_action_index = [prev_action_index]
            action_scores, value, mask_scores, masks, hidden_state = model(
                    stacked_frames, prev_action_index, policy_hidden=hidden_state,
                    device=device)
        elif fusion_model == 'SuperpixelActionConcat':
            # SuperpixelActionConcat's forward will deal with
            # prev_action_features = [None]
            prev_action_features = [prev_action_features]
            (_, value, similarity_scores, actions_masks_features,
                    hidden_state) = model(stacked_frames, prev_action_features,
                            policy_hidden=hidden_state, device=device)
            action_scores = similarity_scores

        # Only attempt one action (which might fail) instead of trying all
        # actions in order
        if sample_action:
            pred_action_index = torch.multinomial(F.softmax(action_scores[0],
                dim=-1), num_samples=1)
        else:
            pred_action_index = torch.argmax(action_scores[0])

        if fusion_model == 'SuperpixelFusion':
            # Only pass mask on interact action so InteractionExploration won't
            # complain
            if (actions[pred_action_index] == constants.ACTIONS_INTERACT or
                    actions[pred_action_index] in constants.INT_ACTIONS):
                if sample_mask:
                    pred_mask_index = torch.multinomial(F.softmax(mask_scores[0],
                        dim=-1), num_samples=1)
                    selected_mask = masks[0][pred_mask_index]
                else:
                    selected_mask = masks[0][torch.argmax(mask_scores[0])]
            else:
                selected_mask = None
            selected_action = actions[pred_action_index]
        elif fusion_model == 'SuperpixelActionConcat':
            selected_action, selected_mask, _ = actions_masks_features[0][
                    pred_action_index]

        if teacher_force:
            selected_action = current_expert_actions[0]['action']
            # TODO: add expert superpixel mask

        frame, reward, done, (action_success, event, err) = (
                env.step(selected_action, interact_mask=selected_mask))
        print(selected_action, action_success, reward, err)
        action_successes.append(action_success)
        all_action_scores.append(action_scores[0])
        values.append(value[0])
        pred_action_indexes.append(pred_action_index)
        rewards.append(reward)
        expert_action_indexes.append(action_to_index[current_expert_actions[0]
            ['action']])
        if fusion_model == 'SuperpixelFusion':
            all_mask_scores.append(mask_scores[0])
            prev_action_index = pred_action_index
        elif fusion_model == 'SuperpixelActionConcat':
            prev_action_features = actions_masks_features[0][pred_action_index][2]
        num_steps += 1

    # Run model one more time to get last value if not done
    if done:
        values.append(torch.zeros(1))
    else:
        frames_with_last = copy.deepcopy(frames)
        frames_with_last.append(torch.from_numpy(np.ascontiguousarray(frame)))

        stacked_frames = stack_frames([frames_with_last[-frame_stack:]],
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
                device=torch.device('cpu'))[0][-1:]
        if fusion_model == 'SuperpixelFusion':
            prev_action_index = [torch.LongTensor([prev_action_index]).to(device)]
            _, value, _, _, _ = model(stacked_frames, prev_action_index,
                    policy_hidden=hidden_state, device=device)
        elif fusion_model == 'SuperpixelActionConcat':
            prev_action_features = [prev_action_features]
            _, value, _, _, _ = model(stacked_frames, prev_action_features,
                    policy_hidden=hidden_state, device=device)
        values.append(value[0])

    print('trajectory len: ' + str(len(all_action_scores)))
    success = done # If all objects are interacted, the episode is a success
    trajectory_results = {}
    trajectory_results['scene_name_or_num'] = env.get_scene_name_or_num()
    trajectory_results['frames'] = frames
    trajectory_results['action_successes'] = action_successes
    trajectory_results['all_action_scores'] = all_action_scores
    trajectory_results['values'] = values
    trajectory_results['pred_action_indexes'] = pred_action_indexes
    trajectory_results['expert_action_indexes'] = expert_action_indexes
    trajectory_results['success'] = float(success)
    trajectory_results['rewards'] = rewards
    # Record average policy entropy over an episode
    # Need to keep grad since these entropies are used for the loss
    action_entropy = per_step_entropy(all_action_scores)
    trajectory_results['action_entropy'] = action_entropy

    if fusion_model == 'SuperpixelFusion':
        trajectory_results['all_mask_scores'] = all_mask_scores
        mask_entropy = per_step_entropy(all_mask_scores)
        trajectory_results['mask_entropy'] = mask_entropy
    return trajectory_results

def train(model, env, optimizer, gamma=1.0, tau=1.0,
        value_loss_coefficient=0.5, entropy_coefficient=0.01, max_grad_norm=50,
        single_interact=False, use_masks=True, fixed_scene_num=None,
        max_trajectory_length=None, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, sample_action=True, sample_mask=True,
        train_episodes=10, valid_seen_episodes=10, valid_unseen_episodes=10,
        eval_interval=1000, max_steps=1000000, device=torch.device('cpu'),
        save_path=None, save_intermediate=False, save_images_video=False,
        load_path=None):
    writer = SummaryWriter(log_dir='tensorboard_logs' if save_path is None else
            os.path.join(save_path, 'tensorboard_logs'))

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
    # If loading from file, metrics will be blank, but that's okay because
    # train_steps and train_frames will be accurate, so it will just pick up
    # where it left off
    last_metrics = {}
    last_metrics['loss'] = []
    last_metrics['policy_loss'] = []
    last_metrics['value_loss'] = []
    last_metrics['success'] = []
    last_metrics['rewards'] = []
    last_metrics['values'] = []
    last_metrics['trajectory_length'] = []
    last_metrics['avg_action_entropy'] = []
    last_metrics['avg_mask_entropy'] = []
    last_metrics['num_masks'] = []
    last_metrics['avg_action_success'] = []
    last_metrics['all_action_scores'] = []
    last_metrics['all_mask_scores'] = []

    # TODO: want a replay memory?
    while train_steps < max_steps:
        # Collect a trajectory
        if fixed_scene_num is not None:
            # If fixed_scene_num is provided, set up that scene exactly the
            # same way every time
            scene_num = fixed_scene_num
            reset_kwargs = {
                    'random_object_positions' : False,
                    'random_position' : False,
                    'random_rotation' : False,
                    'random_look_angle' : False
            }
        else:
            scene_num = random.choice(constants.TRAIN_SCENE_NUMBERS)
            reset_kwargs = {}
        trajectory_results = rollout_trajectory(env, model,
                single_interact=single_interact, use_masks=use_masks,
                max_trajectory_length=max_trajectory_length,
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
                teacher_force=teacher_force, sample_action=sample_action,
                sample_mask=sample_mask, scene_name_or_num=scene_num,
                reset_kwargs=reset_kwargs, device=device)
        all_action_scores = torch.cat(trajectory_results['all_action_scores'])

        # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/train.py
        # https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1).to(device)
        R = trajectory_results['values'][-1]
        rewards = torch.Tensor(trajectory_results['rewards']).to(device)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - trajectory_results['values'][i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = (rewards[i] + gamma *
                trajectory_results['values'][i + 1] -
                trajectory_results['values'][i])

            gae = gae * gamma * tau + delta_t

            chosen_action_index = trajectory_results['pred_action_indexes'][i]
            action_log_prob = trajectory_results['all_action_scores'][i][
                    chosen_action_index]
            action_entropy = trajectory_results['action_entropy'][i]
            mask_entropy = trajectory_results['mask_entropy'][i]
            policy_loss = (policy_loss - action_log_prob * gae -
                    entropy_coefficient * (action_entropy + mask_entropy))

        loss = policy_loss + value_loss_coefficient * value_loss

        optimizer.zero_grad()
        # RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
        try:
            loss.backward(retain_graph=True)
        except:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Compute and save some stats
        train_steps += 1
        train_frames += len(trajectory_results['frames'])
        last_metrics['loss'].append(loss.item())
        last_metrics['policy_loss'].append(policy_loss.item())
        last_metrics['value_loss'].append(value_loss.item())
        last_metrics['success'].append(float(trajectory_results['success']))
        last_metrics['rewards'].append(float(sum(trajectory_results['rewards'])))
        last_metrics['values'].append([value.detach().cpu() for value in
            trajectory_results['values']])
        last_metrics['trajectory_length'].append(
                len(trajectory_results['frames']))
        last_metrics['avg_mask_entropy'].append(
                torch.mean(trajectory_results['mask_entropy']).item())
        last_metrics['avg_action_entropy'].append(
                torch.mean(trajectory_results['action_entropy']).item())
        last_metrics['num_masks'].append(np.mean([len(scores) for scores in
            trajectory_results['all_mask_scores']]))
        last_metrics['avg_action_success'].append(
                np.mean(trajectory_results['action_successes']))
        last_metrics['all_action_scores'].append([action_scores.detach().cpu()
            for action_scores in trajectory_results['all_action_scores']])
        last_metrics['all_mask_scores'].append([mask_scores.detach().cpu() for
            mask_scores in trajectory_results['all_mask_scores']])

        results = {}
        results['train'] = {}
        for metric in last_metrics.keys():
            results['train'][metric] = [last_metrics[metric][-1]]
        # Don't write training results to file
        write_results(writer, results, train_steps, train_frames,
                save_path=None)

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
                    save_path=None)

            # Collect validation statistics and write, print
            # TODO: do we want different max trajectory lengths for eval?
            '''
            results = evaluate(env, model, single_interact=single_interact,
                    use_masks=use_masks, fixed_scene_num=fixed_scene_num,
                    max_trajectory_length=max_trajectory_length,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    train_episodes=train_episodes,
                    valid_seen_episodes=valid_seen_episodes,
                    valid_unseen_episodes=valid_unseen_episodes, device=device)

            write_results(writer, results, train_steps, train_frames,
                    save_path=save_path)
            '''

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
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, checkpoint_save_path)

                if save_images_video:
                    write_images_video(results, train_steps, save_path)

def evaluate(env, model, single_interact=False, use_masks=True,
        fusion_model='SuperpixelFusion', fixed_scene_num=None,
        max_trajectory_length=None, frame_stack=1, zero_fill_frame_stack=False,
        train_episodes=1, valid_seen_episodes=1, valid_unseen_episodes=1,
        device=torch.device('cpu')):
    """
    Evaluate by gathering live rollouts in the environment.
    """
    model.eval()
    metrics = {}
    # Use DATASET_*_SCENE_NUMBERS for now
    for split, episodes, scene_numbers in zip(
            ['train', 'valid_seen', 'valid_unseen'],
            [train_episodes, valid_seen_episodes, valid_unseen_episodes],
            [constants.DATASET_TRAIN_SCENE_NUMBERS,
                constants.DATASET_VALID_SEEN_SCENE_NUMBERS,
                constants.DATASET_VALID_UNSEEN_SCENE_NUMBERS]):
        metrics[split] = {}
        metrics[split]['success'] = []
        metrics[split]['rewards'] = []
        metrics[split]['avg_action_entropy'] = []
        metrics[split]['all_action_scores'] = []
        if fusion_model == 'SuperpixelFusion':
            metrics[split]['all_mask_scores'] = []
            metrics[split]['avg_mask_entropy'] = []
        metrics[split]['trajectory_length'] = []
        metrics[split]['frames'] = []
        metrics[split]['scene_name_or_num'] = []
        for i in range(episodes):
            with torch.no_grad():
                if fixed_scene_num is not None:
                    # If fixed_scene_num is provided, set up that scene exactly the
                    # same way every time
                    scene_num = fixed_scene_num
                    reset_kwargs = {
                            'random_object_positions' : False,
                            'random_position' : False,
                            'random_rotation' : False,
                            'random_look_angle' : False
                    }
                else:
                    scene_num = random.choice(scene_numbers)
                    reset_kwargs = {}
                trajectory_results = rollout_trajectory(env, model,
                        single_interact=single_interact, use_masks=use_masks,
                        fusion_model=fusion_model,
                        max_trajectory_length=max_trajectory_length,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        sample_action=False, sample_mask=False,
                        scene_name_or_num=scene_num,
                        reset_kwargs=reset_kwargs, device=device)
            metrics[split]['success'].append(trajectory_results['success'])
            metrics[split]['rewards'].append(
                    float(sum(trajectory_results['rewards'])))
            metrics[split]['avg_action_entropy'].append(
                    torch.mean(trajectory_results['action_entropy']).item())
            metrics[split]['all_action_scores'].append(
                    [action_scores.detach().cpu() for action_scores in
                        trajectory_results['all_action_scores']])
            if fusion_model == 'SuperpixelFusion':
                metrics[split]['all_mask_scores'].append(
                        [mask_scores.detach().cpu() for mask_scores in
                            trajectory_results['all_mask_scores']])
                metrics[split]['avg_mask_entropy'].append(
                        torch.mean(trajectory_results['mask_entropy']).item())
            metrics[split]['trajectory_length'].append(
                    float(len(trajectory_results['frames'])))
            metrics[split]['frames'].append(trajectory_results['frames'])
            metrics[split]['scene_name_or_num'].append(
                    trajectory_results['scene_name_or_num'])

    model.train()
    #model.visual_model.resnet_model.model.eval()
    return metrics

def write_results(writer, results, train_steps, train_frames,
        train_trajectories=None, save_path=None):
    """
    Write results to SummaryWriter.
    """
    for split in results.keys():
        for metric, values in results[split].items():
            steps_name = 'steps/' + split + '/' + metric
            frames_name = 'frames/' + split + '/' + metric
            trajectories_name = 'trajectories/' + split + '/' + metric

            if metric in ['frames', 'scene_name_or_num']:
                continue
            # all_mask_scores can be variably sized so it won't work with
            # histogram unless we take the top k, but in that case entropy
            # works fine as a measure
            if metric == 'all_action_scores':
                # all_action_scores is a list of lists (for each trajectory) of
                # tensors (for each step)
                trajectory_flat_action_scores = []
                for action_scores in values:
                    trajectory_flat_action_scores.extend(action_scores)
                writer.add_histogram(steps_name,
                        torch.stack(trajectory_flat_action_scores),
                        train_steps)
                writer.add_histogram(frames_name,
                        torch.stack(trajectory_flat_action_scores),
                        train_frames)

                # Add per-action score histograms
                for action_i in range(len(values[0][0])):
                    action_name = (constants.SIMPLE_ACTIONS[action_i] if
                            len(values[0][0]) == len(constants.SIMPLE_ACTIONS)
                            else constants.COMPLEX_ACTIONS[action_i])
                    steps_name_action_i = steps_name + '_' + action_name
                    frames_name_action_i = frames_name + '_' + action_name
                    trajectories_name_action_i = (trajectories_name + '_' +
                            action_name)

                    flat_action_i_scores = []
                    for trajectory in range(len(values)):
                        flat_action_i_scores.extend([action_scores[action_i]
                            for action_scores in values[trajectory]])

                    writer.add_histogram(steps_name_action_i,
                            torch.stack(flat_action_i_scores), train_steps)
                    writer.add_histogram(frames_name_action_i,
                            flat_action_i_scores, train_frames)
                    if train_trajectories is not None:
                        writer.add_histogram(trajectories_name_action_i,
                                flat_action_i_scores, train_trajectories)

                # Also histogram the top (chosen) action
                action_argmaxes = [torch.argmax(action_scores).item() for
                        action_scores in trajectory_flat_action_scores]
                action_argmaxes_steps_name = ('steps/' + split + '/' +
                        'action_argmaxes')
                action_argmaxes_frames_name = ('frames/' + split + '/' +
                        'action_argmaxes')
                action_argmaxes_trajectories_name = ('trajectories/' + split +
                        '/' + 'action_argmaxes')
                writer.add_histogram(action_argmaxes_steps_name, action_argmaxes,
                        train_steps, bins='fd')
                writer.add_histogram(action_argmaxes_frames_name,
                        action_argmaxes, train_frames, bins='fd')
                if train_trajectories is not None:
                    writer.add_histogram(trajectories_name, action_argmaxes,
                            train_trajectories)
                    writer.add_histogram(action_argmaxes_trajectories_name,
                            action_argmaxes, train_trajectories, bins='fd')
            elif 'scores' in metric: # Skip all_mask_scores
                continue
            elif metric == 'values':
                # Values is a list of lists (for each trajectory) of scalars
                flat_value_scores = []
                for value_scores in values:
                    flat_value_scores.extend(value_scores)
                writer.add_histogram(steps_name,
                        torch.stack(flat_value_scores),
                        train_steps)
                writer.add_histogram(frames_name,
                        torch.stack(flat_value_scores),
                        train_frames)

                # Add per-step state-value histograms
                for i in range(len(values[0])):
                    steps_name_i = steps_name + '_' + str(i)
                    frames_name_i = frames_name + '_' + str(i)
                    trajectories_name_i = trajectories_name + '_' + str(i)

                    writer.add_histogram(steps_name_i,
                            torch.stack([value_scores[i] for value_scores in
                                values]), train_steps)
                    writer.add_histogram(frames_name_i,
                            torch.stack([value_scores[i] for value_scores in
                                values]), train_frames)
                    if train_trajectories is not None:
                        writer.add_histogram(trajectories_name_i,
                                torch.stack([value_scores[i] for value_scores in
                                    values]), train_trajectories)
                avg_value = torch.mean(torch.stack(flat_value_scores)).item()
                avg_value_steps_name = ('steps/' + split + '/' +
                        'avg_value')
                avg_value_frames_name = ('frames/' + split + '/' +
                        'avg_value')
                avg_values_trajectories_name = ('trajectories/' + split +
                        '/' + 'avg_value')
                writer.add_scalar(avg_value_steps_name, avg_value, train_steps)
                writer.add_scalar(avg_value_frames_name, avg_value,
                        train_frames)
                if train_trajectories is not None:
                    writer.add_histogram(trajectories_name,
                            torch.stack(flat_value_scores), train_trajectories)
                    writer.add_scalar(avg_value_trajectories_name, avg_value,
                            train_trajectories)
            else:
                mean = np.mean(values)
                writer.add_scalar(steps_name, mean, train_steps)
                writer.add_scalar(frames_name, mean, train_frames)
                if train_trajectories is not None:
                    writer.add_scalar(trajectories_name, mean,
                            train_trajectories)

    # Also write output to saved file
    if save_path is not None:
        results_path = os.path.join(save_path, str(train_steps))
        # Exclude frames from results
        json_results = {}
        for split in results.keys():
            # Don't save scores
            json_results[split] = {k:v for k, v in results[split].items() if k
                    != 'frames' and 'scores' not in k}
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        with open(os.path.join(results_path, 'results.json'), 'w') as \
                jsonfile:
            json.dump(json_results, jsonfile)

def write_images_video(results_online, train_steps, save_path):
    for split in results_online.keys():
        for trajectory_index in range(len(results_online[split]['frames'])):
            trajectory_images = results_online[split] \
                    ['frames'][trajectory_index]
            scene_name_or_num = str(results_online[split]['scene_name_or_num']
                    [trajectory_index])
            success_or_failure = 'success' if results_online[split] \
                    ['success'][trajectory_index] else 'failure'
            reward = str(sum(results_online[split]['rewards']
                [trajectory_index]))

            trajectory_length = str(results_online[split]['trajectory_length']
                    [trajectory_index])
            images_save_path =  os.path.join(save_path, 'images_video',
                    str(train_steps), split, str(trajectory_index) + '_' +
                    'scene' + scene_name_or_num + '_' + success_or_failure +
                    '_' + 'reward' + reward + 'trajectorylen' +
                    trajectory_length)
            if not os.path.isdir(images_save_path):
                os.makedirs(images_save_path)

            for image_index in range(len(trajectory_images)):
                image_save_path = os.path.join(images_save_path, '%09d.png' %
                        int(image_index))
                cv2.imwrite(image_save_path, trajectory_images[image_index].numpy())
            video_save_path = os.path.join(images_save_path,
                    'video.mp4')
            video_saver.save(os.path.join(images_save_path,
                '*.png'), video_save_path)


if __name__ == '__main__':
    # TODO: make this setup code a function somewhere
    import torch.nn as nn
    import torch.optim as optim

    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
    from env.thor_env import ThorEnv
    from env.reward import InteractionReward
    from models.nn.resnet import Resnet
    from models.nn.ie import LSTMPolicy, SuperpixelFusion

    args = parse_args()

    thor_env = ThorEnv()

    with open(args.reward_config_path, 'r') as jsonfile:
        reward_config = json.load(jsonfile)['InteractionExploration']

    # TODO: if adding threads, add support for running on multiple gpus, e.g.
    # gpu_ids like
    # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/train.py
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    reward = InteractionReward(thor_env, reward_config,
            reward_rotations_look_angles=args.reward_rotations_look_angles,
            reward_state_changes=args.reward_state_changes,
            persist_state=args.reward_persist_state,
            repeat_discount=args.reward_repeat_discount)

    ie = InteractionExploration(thor_env, reward,
            single_interact=args.single_interact,
            sample_contextual_action=args.sample_contextual_action,
            use_masks=args.use_masks)

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    slic_kwargs = {
            'max_iter' : args.slic_max_iter,
            'spacing' : None,
            'multichannel' : True,
            'convert2lab' : True,
            'enforce_connectivity' : True,
            'max_size_factor' : args.slic_max_size_factor,
            'n_segments' : args.slic_n_segments,
            'compactness' : args.slic_compactness,
            'sigma' : args.slic_sigma,
            'min_size_factor' : args.slic_min_size_factor
    }

    if args.single_interact:
        num_actions = len(constants.SIMPLE_ACTIONS)
    else:
        num_actions = len(constants.COMPLEX_ACTIONS)

    action_embeddings = nn.Embedding(num_embeddings=num_actions,
            embedding_dim=args.action_embedding_dim).to(device)

    if 'resnet' in args.visual_model:
        args.visual_feature_size = 512
        resnet_args = Namespace(visual_model='resnet', gpu=args.gpu >= 0,
                gpu_index=args.gpu)
        visual_model = Resnet(resnet_args, use_conv_feat=False)
    else:
        print("visual model '" + args.visual_model + "' not supported")

    if 'resnet' in args.superpixel_model:
        args.superpixel_feature_size = 512
        resnet_args = Namespace(visual_model='resnet', gpu=args.gpu >= 0,
                gpu_index=args.gpu)
        superpixel_model = Resnet(resnet_args, use_conv_feat=False)
    else:
        print("superpixel model '" + args.superpixel_model + "' not supported")

    if type(args.action_fc_units) is int:
        args.action_fc_units = [args.action_fc_units]
    if type(args.visual_fc_units) is int:
        args.visual_fc_units = [args.visual_fc_units]

    policy_model = LSTMPolicy(num_actions=num_actions,
            visual_feature_size=args.visual_feature_size,
            superpixel_feature_size=args.superpixel_feature_size,
            prev_action_size=args.action_embedding_dim,
            lstm_hidden_size=args.lstm_hidden_dim, dropout=args.dropout,
            action_fc_units=args.action_fc_units,
            value_fc_units=args.value_fc_units,
            visual_fc_units=args.visual_fc_units,
            prev_action_after_lstm=args.prev_action_after_lstm).to(device)

    sf = SuperpixelFusion(action_embeddings=action_embeddings,
          visual_model=visual_model, superpixel_model=superpixel_model,
          policy_model=policy_model, slic_kwargs=slic_kwargs,
          boundary_pixels=args.boundary_pixels,
          neighbor_depth=args.neighbor_depth,
          neighbor_connectivity=args.neighbor_connectivity, black_outer=False,
          device=device)
    try:
        sf = sf.to(device)
    except:
        sf = sf.to(device)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(sf.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(sf.parameters(), lr=args.lr)
    if 'adam' in args.optimizer:
        amsgrad = 'amsgrad' in args.optimizer
        optimizer = optim.Adam(sf.parameters(), lr=args.lr, amsgrad=amsgrad)

    print('model parameters: ' + str(sum(p.numel() for p in sf.parameters()
        if p.requires_grad)))

    train(sf, ie, optimizer, gamma=args.gamma, tau=args.tau,
            value_loss_coefficient=args.value_loss_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            max_grad_norm=args.max_grad_norm,
            single_interact=args.single_interact, use_masks=args.use_masks,
            fixed_scene_num=args.fixed_scene_num,
            max_trajectory_length=args.max_trajectory_length,
            frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            teacher_force=args.teacher_force, sample_action=args.sample_action,
            sample_mask=args.sample_mask, train_episodes=args.train_episodes,
            valid_seen_episodes=args.valid_seen_episodes,
            valid_unseen_episodes=args.valid_unseen_episodes,
            eval_interval=args.eval_interval, max_steps=args.max_steps,
            device=device, save_path=args.save_path, load_path=args.load_path)

