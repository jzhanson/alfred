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

def rollout_trajectory(env, model, single_interact=False, use_masks=True,
        max_trajectory_length=None, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, scene_name_or_num=None,
        device=torch.device('cpu')):
    """
    Returns dictionary of trajectory results.
    """
    frames = []
    all_action_scores = []
    all_mask_scores = []
    values = []
    pred_action_indexes = []
    rewards = []
    expert_action_indexes = []
    frame = env.reset(scene_name_or_num)
    done = False
    num_steps = 0

    actions = (constants.SIMPLE_ACTIONS if single_interact else
            constants.COMPLEX_ACTIONS)
    action_to_index = (constants.ACTION_TO_INDEX_SIMPLE if single_interact else
            constants.ACTION_TO_INDEX_COMPLEX)

    prev_action_index = None
    hidden_state = model.init_policy_hidden(batch_size=1, device=device)
    while not done and (max_trajectory_length is None or num_steps <
            max_trajectory_length):
        frames.append(torch.from_numpy(np.ascontiguousarray(frame)))
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
        if prev_action_index is not None:
            prev_action_index = torch.LongTensor([prev_action_index]).to(device)
        else:
            # Let SuperpixelFusion's forward take care of this
            prev_action_index = [prev_action_index]
        action_scores, value, mask_scores, masks, hidden_state = model(
                stacked_frames, prev_action_index, policy_hidden=hidden_state,
                device=device)
        # Sorted in increasing order (rightmost is highest scoring action)
        sorted_scores, top_indices = torch.sort(action_scores[0],
                descending=True)
        top_indices = top_indices.flatten()
        # Try each action until success
        pred_action_index = None
        # TODO: add action sampling and mask sampling
        #pred_action_index = torch.multinomial(F.softmax(action_scores[0]), num_samples=1)
        current_expert_actions, _ = env.get_current_expert_actions_path()
        for i in range(len(actions)):
            pred_action_index = top_indices[i]
            # Only pass mask on interact action so InteractionExploration won't
            # complain
            if (actions[pred_action_index] == constants.ACTIONS_INTERACT or
                    actions[pred_action_index] in constants.INT_ACTIONS):
                # TODO: try each mask?
                selected_mask = masks[0][torch.argmax(mask_scores[0])]
            else:
                selected_mask = None
            if teacher_force:
                selected_action = current_expert_actions[0]['action']
                # TODO: add expert superpixel mask
            else:
                selected_action = actions[pred_action_index]
            frame, reward, done, (action_success, event, err) = (
                    env.step(selected_action, interact_mask=selected_mask))
            if action_success:
                print(selected_action)
                # TODO: consider penalizing failed actions more or only allow
                # one failed action
                # TODO: report failed actions
                # TODO: keep track of chosen action distribution with
                # tensorboard
                break
        assert pred_action_index is not None
        all_action_scores.append(action_scores[0])
        values.append(value[0])
        pred_action_indexes.append(pred_action_index)
        rewards.append(reward)
        expert_action_indexes.append(action_to_index[current_expert_actions[0]
            ['action']])
        all_mask_scores.append(mask_scores[0])
        prev_action_index = pred_action_index
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
        prev_action_index = torch.LongTensor([prev_action_index]).to(device)
        _, value, _, _, _ = model(stacked_frames, prev_action_index,
                policy_hidden=hidden_state, device=device)
        values.append(value[0])

    print('trajectory len: ' + str(len(all_action_scores)))
    success = done # If all objects are interacted, the episode is a success
    trajectory_results = {}
    trajectory_results['scene_name_or_num'] = env.get_scene_name_or_num()
    trajectory_results['frames'] = frames
    trajectory_results['all_action_scores'] = all_action_scores
    trajectory_results['all_mask_scores'] = all_mask_scores
    trajectory_results['values'] = values
    trajectory_results['pred_action_indexes'] = pred_action_indexes
    trajectory_results['expert_action_indexes'] = expert_action_indexes
    trajectory_results['success'] = float(success)
    trajectory_results['rewards'] = rewards
    # Record average policy entropy over an episode
    with torch.no_grad():
        mask_entropy = per_step_entropy(all_mask_scores)
        action_entropy = per_step_entropy(all_action_scores)
    trajectory_results['action_entropy'] = action_entropy
    trajectory_results['mask_entropy'] = mask_entropy
    return trajectory_results

def train(model, env, optimizer, gamma=1.0, tau=1.0,
        value_loss_coefficient=0.5, entropy_coefficient=0.01, max_grad_norm=50,
        single_interact=False, use_masks=True, max_trajectory_length=None,
        frame_stack=1, zero_fill_frame_stack=False, teacher_force=False,
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
    # TODO: save/load metrics instead of relying on tensorboard and make metric
    # loading work well with tensorboard
    last_metrics = {}
    last_metrics['loss'] = []
    last_metrics['success'] = []
    last_metrics['rewards'] = []
    last_metrics['trajectory_length'] = []
    last_metrics['avg_action_entropy'] = []
    last_metrics['avg_mask_entropy'] = []
    last_metrics['num_masks'] = []

    # TODO: want a replay memory?
    while train_steps < max_steps:
        # Collect a trajectory
        trajectory_results = rollout_trajectory(env, model,
                single_interact=single_interact, use_masks=use_masks,
                max_trajectory_length=max_trajectory_length,
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
                teacher_force=teacher_force,
                scene_name_or_num=random.choice(
                    constants.TRAIN_SCENE_NUMBERS), device=device)
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
        last_metrics['success'].append(float(trajectory_results['success']))
        last_metrics['rewards'].append(float(sum(trajectory_results['rewards'])))
        last_metrics['trajectory_length'].append(
                len(trajectory_results['frames']))
        last_metrics['avg_mask_entropy'].append(
                torch.mean(trajectory_results['mask_entropy']).item())
        last_metrics['avg_action_entropy'].append(
                torch.mean(trajectory_results['action_entropy']).item())
        last_metrics['num_masks'].append(np.mean([len(scores) for scores in
            trajectory_results['all_mask_scores']]))

        results = {}
        results['train'] = {}
        for metric in last_metrics.keys():
            results['train'][metric] = last_metrics[metric][-1]
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
            results = evaluate(env, model, single_interact=single_interact,
                    use_masks=use_masks,
                    max_trajectory_length=max_trajectory_length,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    train_episodes=train_episodes,
                    valid_seen_episodes=valid_seen_episodes,
                    valid_unseen_episodes=valid_unseen_episodes, device=device)

            write_results(writer, results, train_steps, train_frames,
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
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, checkpoint_save_path)

                if save_images_video:
                    write_images_video(results, train_steps, save_path)

def evaluate(env, model, single_interact=False, use_masks=True,
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
        metrics[split]['avg_mask_entropy'] = []
        metrics[split]['trajectory_length'] = []
        metrics[split]['frames'] = []
        metrics[split]['scene_name_or_num'] = []
        for i in range(episodes):
            with torch.no_grad():
                trajectory_results = rollout_trajectory(env, model,
                        single_interact=single_interact, use_masks=use_masks,
                        max_trajectory_length=max_trajectory_length,
                        frame_stack=frame_stack,
                        zero_fill_frame_stack=zero_fill_frame_stack,
                        scene_name_or_num=random.choice(scene_numbers),
                        device=device)
            # TODO: append in a loop over keys here and elsewhere
            metrics[split]['success'].append(trajectory_results['success'])
            metrics[split]['rewards'].append(
                    float(sum(trajectory_results['rewards'])))
            metrics[split]['avg_action_entropy'].append(
                    torch.mean(trajectory_results['action_entropy']).item())
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
            if metric in ['frames', 'scene_name_or_num']:
                continue
            mean = np.mean(values)
            writer.add_scalar('steps/' + split + '/' + metric, mean,
                    train_steps)
            writer.add_scalar('frames/' + split + '/' + metric, mean,
                    train_frames)
            if train_trajectories is not None:
                writer.add_scalar('trajectories/' + split + '/' + metric, mean,
                        train_trajectories)
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

    # TODO: add support for running on multiple gpus, e.g. gpu_ids like
    # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/train.py
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    reward = InteractionReward(thor_env, reward_config,
            reward_rotations=args.reward_rotations,
            reward_look_angle=args.reward_look_angle,
            reward_state_changes=args.reward_state_changes)

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
            max_trajectory_length=args.max_trajectory_length,
            frame_stack=args.frame_stack,
            zero_fill_frame_stack=args.zero_fill_frame_stack,
            teacher_force=args.teacher_force,
            train_episodes=args.train_episodes,
            valid_seen_episodes=args.valid_seen_episodes,
            valid_unseen_episodes=args.valid_unseen_episodes,
            eval_interval=args.eval_interval, max_steps=args.max_steps,
            device=device, save_path=args.save_path, load_path=args.load_path)

