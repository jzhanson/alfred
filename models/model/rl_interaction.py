import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import random
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import gen.constants as constants
from models.utils.metric import per_step_entropy, trajectory_avg_entropy
from models.utils.helper_utils import (stack_frames,
        superpixelactionconcat_get_num_superpixels,
        superpixelactionconcat_index_to_action)
from utils.video_util import VideoSaver

video_saver = VideoSaver()

from tensorboardX import SummaryWriter

"""
There's a "bug" where the underlying Unity environment doesn't like being
passed objects for ToggleObjectOn/Off that aren't visible and throws an
exception that's caught by env/thor_env.py, but seems to work fine for other
interactions with not visible objects. This is a small issue since the
"visible" distance is not very large in the THOR environment.
"""

def rollout_trajectory(env, model, single_interact=False, use_masks=True,
        use_gt_segmentation=False, fusion_model='SuperpixelFusion',
        outer_product_sampling=False, inverse_score=False,
        zero_null_superpixel_features=True, curiosity_model=None,
        max_trajectory_length=None, frame_stack=1, zero_fill_frame_stack=False,
        teacher_force=False, sample_action=True, sample_mask=True,
        scene_name_or_num=None, reset_kwargs={}, device=torch.device('cpu')):
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
        if outer_product_sampling:
            # Keep track of logits for interaction actions also instead of just
            # individual (action x superpixel) softmax scores
            discrete_action_logits = []
        else:
            pred_mask_indexes = []
    if curiosity_model is not None:
        curiosity_rewards = []
        curiosity_losses = []
    frame = env.reset(scene_name_or_num, **reset_kwargs)
    done = False
    num_steps = 0
    prev_action_features = torch.zeros(1, model.policy_model.prev_action_size,
            device=device)

    actions = (constants.SIMPLE_ACTIONS if single_interact else
            constants.COMPLEX_ACTIONS)
    action_to_index = (constants.ACTION_TO_INDEX_SIMPLE if single_interact else
            constants.ACTION_TO_INDEX_COMPLEX)

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

        if use_gt_segmentation:
            # Get ThorEnv inside InteractionExploration env
            gt_segmentation = env.env.last_event.instance_segmentation_frame
        else:
            gt_segmentation = None

        if fusion_model == 'SuperpixelFusion':
            action_scores, value, mask_scores, masks, (action_features,
                    mask_features), hidden_state = model(
                    stacked_frames, prev_action_features,
                    policy_hidden=hidden_state,
                    gt_segmentation=gt_segmentation, device=device)
            # Masks scores are dot product similarities, so we can use
            # inverse_score (action scores are logits)
            # We need to change mask_scores which are reported in
            # trajectory_results for the action_log_prob calculation
            if inverse_score:
                mask_scores = [1 / torch.sigmoid(mask_scores[0])]
            # Only attempt one action (which might fail) instead of trying all
            # actions in order
            if outer_product_sampling:
                # Only outer product interaction actions with masks, after
                # softmaxing both actions and masks so when we concatenate
                # navigation actions and (interaction actions x masks) the
                # probability distribution is valid
                actions_softmax = F.softmax(action_scores, dim=-1)
                # Keep batch dimension for consistency
                masks_softmax = [F.softmax(mask_scores[0], dim=-1)]

                # First dimension is actions, second dimension is masks.
                # torch.outer (torch.ger in 1.1.0) wants two 1D vectors
                outer_scores = torch.ger(
                        actions_softmax[0][len(constants.NAV_ACTIONS):],
                        masks_softmax[0])
                # Still keep batch dimension
                concatenated_softmax = [torch.cat([
                    actions_softmax[0][:len(constants.NAV_ACTIONS)],
                    torch.flatten(outer_scores)])]
                if sample_action:
                    pred_action_index = torch.multinomial(
                            concatenated_softmax[0], num_samples=1)
                else:
                    # Unsqueeze to make pred_action_index 1-D tensor to match
                    # sampling case
                    pred_action_index = torch.argmax(
                            concatenated_softmax[0]).unsqueeze(0)

                if pred_action_index < len(constants.NAV_ACTIONS):
                    selected_action = actions[pred_action_index]
                    pred_mask_index = None
                    selected_mask = None
                else:
                    selected_action = superpixelactionconcat_index_to_action(
                            pred_action_index, len(concatenated_softmax[0]),
                            single_interact=single_interact)
                    pred_mask_index = (pred_action_index -
                            len(constants.NAV_ACTIONS)) % len(masks[0])
                    selected_mask = masks[0][pred_mask_index]
            else:
                if sample_action:
                    pred_action_index = torch.multinomial(F.softmax(
                        action_scores[0], dim=-1), num_samples=1)
                else:
                    pred_action_index = torch.argmax(
                            action_scores[0]).unsqueeze(0)
                # Also sample a mask - only on interact action so
                # InteractionExploration won't complain
                if (actions[pred_action_index] == constants.ACTIONS_INTERACT or
                        actions[pred_action_index] in constants.INT_ACTIONS):
                    if sample_mask:
                        pred_mask_index = torch.multinomial(F.softmax(
                            mask_scores[0], dim=-1), num_samples=1)
                    else:
                        pred_mask_index = torch.argmax(
                                mask_scores[0]).unsqueeze(0)
                    selected_mask = masks[0][pred_mask_index]
                else:
                    pred_mask_index = None
                    selected_mask = None
                selected_action = actions[pred_action_index]

            # Construct prev_action_features
            if selected_mask is None:
                if zero_null_superpixel_features:
                    null_mask_features = torch.zeros_like(mask_features[0][0],
                            device=device)
                else:
                    null_mask_features = torch.mean(mask_features[0], dim=-1)
                prev_action_features = torch.cat([
                    action_features[0][action_to_index[selected_action]],
                    null_mask_features])
            else:
                # pred_mask_index is a 1D tensor, which makes tensor indexing
                # include the indexed dimension (so the shape is (1, 512)), so
                # we have to index it to get the 0D tensor so the indexed
                # tensor's shape is (512)
                prev_action_features = torch.cat([
                    action_features[0][action_to_index[selected_action]],
                    mask_features[0][pred_mask_index[0]]])
            prev_action_features = torch.unsqueeze(prev_action_features, 0)
        elif fusion_model == 'SuperpixelActionConcat':
            (_, value, similarity_scores, actions_masks_features,
                    hidden_state) = model(stacked_frames, prev_action_features,
                            policy_hidden=hidden_state,
                            gt_segmentation=gt_segmentation, device=device)
            action_scores = similarity_scores
            # Same deal - we need to change action_scores which will be
            # reported via trajectory_results so the action log
            # probability is correct (post-sigmoid)
            if inverse_score:
                action_scores = [1 / torch.sigmoid(action_scores[0])]
            if sample_action:
                pred_action_index = torch.multinomial(F.softmax(
                    action_scores[0], dim=-1), num_samples=1)
            else:
                pred_action_index = torch.argmax(action_scores[0]).unsqueeze(0)

            selected_action, selected_mask, _ = actions_masks_features[0][
                    pred_action_index]

            prev_action_features = actions_masks_features[0][
                    pred_action_index][2]
            prev_action_features = torch.unsqueeze(prev_action_features, 0)

        if teacher_force:
            # selected_action and therefore action_success will not match
            # pred_action_index if teacher forcing!
            selected_action = current_expert_actions[0]['action']
            # TODO: add expert superpixel mask

        next_frame, reward, done, (action_success, event, err) = (
                env.step(selected_action, interact_mask=selected_mask))

        if curiosity_model is not None:
            next_stacked_frames = stack_frames([frames[-frame_stack+1:] +
                [torch.from_numpy(np.ascontiguousarray(next_frame))]],
                frame_stack=frame_stack,
                zero_fill_frame_stack=zero_fill_frame_stack,
                device=torch.device('cpu'))[0][-1:]
            curiosity_reward, curiosity_loss = curiosity_model(stacked_frames,
                    prev_action_features, next_stacked_frames)
            curiosity_rewards.append(curiosity_reward)
            curiosity_losses.append(curiosity_loss)
            print(selected_action, action_success, reward,
                    curiosity_reward.item(), err)
        else:
            print(selected_action, action_success, reward, err)

        frame = next_frame
        action_successes.append(action_success)
        values.append(value[0])
        pred_action_indexes.append(pred_action_index)
        rewards.append(reward)
        expert_action_indexes.append(action_to_index[current_expert_actions[0]
            ['action']])
        if fusion_model == 'SuperpixelFusion':
            all_mask_scores.append(mask_scores[0])
            if outer_product_sampling:
                # We have to use the softmax scores here because there isn't a
                # unified actions+masks score. Fortunately, softmax is
                # differentiable
                all_action_scores.append(concatenated_softmax[0])
                discrete_action_logits.append(action_scores[0])
            else:
                pred_mask_indexes.append(pred_mask_index)
                all_action_scores.append(action_scores[0])
        elif fusion_model == 'SuperpixelActionConcat':
            all_action_scores.append(action_scores[0])
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
            _, value, _, _, _, _ = model(stacked_frames, prev_action_features,
                    policy_hidden=hidden_state, device=device)
        elif fusion_model == 'SuperpixelActionConcat':
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
        if outer_product_sampling:
            trajectory_results['discrete_action_logits'] = (
                    discrete_action_logits)
        else:
            trajectory_results['pred_mask_indexes'] = pred_mask_indexes
    if curiosity_model is not None:
        trajectory_results['curiosity_rewards'] = curiosity_rewards
        trajectory_results['curiosity_losses'] = curiosity_losses
    return trajectory_results

def train(model, env, optimizer, gamma=1.0, tau=1.0,
        value_loss_coefficient=0.5, entropy_coefficient=0.01, max_grad_norm=50,
        single_interact=False, use_masks=True, use_gt_segmentation=False,
        fusion_model='SuperpixelFusion', outer_product_sampling=False,
        inverse_score=False, zero_null_superpixel_features=True,
        curiosity_model=None, curiosity_lambda=0.1, scene_numbers=None,
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
    last_metrics['num_masks'] = []
    last_metrics['action_successes'] = []
    last_metrics['pred_action_indexes'] = []
    last_metrics['all_action_scores'] = []
    if fusion_model == 'SuperpixelFusion':
        last_metrics['all_mask_scores'] = []
        last_metrics['avg_mask_entropy'] = []
        if outer_product_sampling:
            last_metrics['discrete_action_logits'] = []
    if curiosity_model is not None:
        last_metrics['curiosity_rewards'] = []
        last_metrics['avg_curiosity_loss'] = []

    # TODO: want a replay memory?
    while train_steps < max_steps:
        # Collect a trajectory
        if len(scene_numbers) == 1:
            # If only one scene_number is provided, set up that scene exactly
            # the same way every time
            scene_num = scene_numbers[0]
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
                use_gt_segmentation=use_gt_segmentation,
                fusion_model=fusion_model,
                outer_product_sampling=outer_product_sampling,
                inverse_score=inverse_score,
                zero_null_superpixel_features=zero_null_superpixel_features,
                curiosity_model=curiosity_model,
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
        if curiosity_model is not None:
            rewards += torch.stack(trajectory_results['curiosity_rewards'])
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            # TODO: option to normalize advantages? PPO?
            advantage = R - trajectory_results['values'][i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = (rewards[i] + gamma *
                trajectory_results['values'][i + 1] -
                trajectory_results['values'][i])

            gae = gae * gamma * tau + delta_t

            chosen_action_index = trajectory_results['pred_action_indexes'][i]
            if outer_product_sampling:
                # Scores are already softmaxed and have action and mask
                # combined
                action_log_prob = torch.log(
                        trajectory_results['all_action_scores'][i][
                                chosen_action_index])
            else:
                action_log_prob = F.log_softmax(
                        trajectory_results['all_action_scores'][i], dim=-1)[
                                chosen_action_index]
                if fusion_model == 'SuperpixelFusion':
                    chosen_mask_index = trajectory_results[
                            'pred_mask_indexes'][i]
                    if chosen_mask_index is not None:
                        mask_log_prob = F.log_softmax(
                            trajectory_results['all_mask_scores'][i], dim=-1)[
                                    chosen_mask_index]
                        action_log_prob += mask_log_prob
            action_entropy = trajectory_results['action_entropy'][i]
            if fusion_model == 'SuperpixelFusion':
                mask_entropy = trajectory_results['mask_entropy'][i]
            elif fusion_model == 'SuperpixelActionConcat':
                # We only need to apply entropy once over concatenated
                # actions+masks
                mask_entropy = 0
            policy_loss = (policy_loss - action_log_prob * gae -
                    entropy_coefficient * (action_entropy + mask_entropy))

        if curiosity_model is not None:
            curiosity_loss = torch.mean(torch.stack(
                trajectory_results['curiosity_losses']))
            loss = curiosity_lambda * (policy_loss + value_loss_coefficient *
                    value_loss) + curiosity_loss
        else:
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
        last_metrics['avg_action_entropy'].append(
                torch.mean(trajectory_results['action_entropy']).item())
        last_metrics['action_successes'].append(
                trajectory_results['action_successes'])
        last_metrics['pred_action_indexes'].append(
                [pred_action_index.item() for pred_action_index in
                    trajectory_results['pred_action_indexes']])
        last_metrics['all_action_scores'].append([action_scores.detach().cpu()
            for action_scores in trajectory_results['all_action_scores']])
        if fusion_model == 'SuperpixelFusion':
            last_metrics['num_masks'].append(np.mean([len(scores) for scores in
                trajectory_results['all_mask_scores']]))
            last_metrics['all_mask_scores'].append([mask_scores.detach().cpu() for
                mask_scores in trajectory_results['all_mask_scores']])
            last_metrics['avg_mask_entropy'].append(
                    torch.mean(trajectory_results['mask_entropy']).item())
            if outer_product_sampling:
                last_metrics['discrete_action_logits'].append([logits.detach().cpu()
                    for logits in
                    trajectory_results['discrete_action_logits']])
        elif fusion_model == 'SuperpixelActionConcat':
            last_metrics['num_masks'].append(np.mean([(len(scores) -
                len(constants.NAV_ACTIONS)) / (1 if single_interact else
                    len(constants.INT_ACTIONS)) for scores in
                trajectory_results['all_action_scores']]))
        if curiosity_model is not None:
            last_metrics['curiosity_rewards'].append(
                    torch.sum(torch.stack(
                        trajectory_results['curiosity_rewards'])).item())
            last_metrics['avg_curiosity_loss'].append(
                    torch.mean(torch.stack(
                        trajectory_results['curiosity_losses'])).item())

        results = {}
        results['train'] = {}
        for metric in last_metrics.keys():
            results['train'][metric] = [last_metrics[metric][-1]]
        # Don't write training results to file, and only graph train_steps (no
        # need for train_frames or train_trajectories to save on tensorboard
        # file size)
        write_results(writer, results, train_steps, fusion_model=fusion_model,
                outer_product_sampling=outer_product_sampling,
                single_interact=single_interact, save_path=None)

        # Evaluate and save checkpoint every N trajectories, collect/print
        # stats
        if train_steps % eval_interval == 0 or train_steps == max_steps:
            print('steps %d frames %d' % (train_steps, train_frames))
            results = {}
            results['train'] = {}
            for metric, values in last_metrics.items():
                results['train']['avg/' + metric] = values
                last_metrics[metric] = []
            '''
            write_results(writer, results, train_steps,
                    fusion_model=fusion_model,
                    outer_product_sampling=outer_product_sampling,
                    single_interact=single_interact,
                    save_path=None)
            '''

            # Collect validation statistics and write, print
            # TODO: do we want different max trajectory lengths, scene numbers
            # for eval?
            '''
            results = evaluate(env, model, single_interact=single_interact,
                    use_masks=use_masks,
                    use_gt_segmentation=use_gt_segmentation,
                    fusion_model=fusion_model,
                    outer_product_sampling=outer_product_sampling,
                    zero_null_superpixel_features=zero_null_superpixel_features,
                    scene_numbers=scene_numbers,
                    max_trajectory_length=max_trajectory_length,
                    frame_stack=frame_stack,
                    zero_fill_frame_stack=zero_fill_frame_stack,
                    train_episodes=train_episodes,
                    valid_seen_episodes=valid_seen_episodes,
                    valid_unseen_episodes=valid_unseen_episodes, device=device)

            write_results(writer, results, train_steps,
                    fusion_model=fusion_model,
                    outer_product_sampling=outer_product_sampling,
                    single_interact=single_interact, save_path=save_path)
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

# TODO: update scene_numbers argument and below
def evaluate(env, model, single_interact=False, use_masks=True,
        use_gt_segmentation=False, fusion_model='SuperpixelFusion',
        outer_product_sampling=False, inverse_score=False,
        zero_null_superpixel_features=True, scene_numbers=None,
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
        # TODO: add other metrics that are in last_metrics
        metrics[split]['success'] = []
        metrics[split]['rewards'] = []
        metrics[split]['avg_action_entropy'] = []
        metrics[split]['all_action_scores'] = []
        if fusion_model == 'SuperpixelFusion':
            metrics[split]['all_mask_scores'] = []
            metrics[split]['avg_mask_entropy'] = []
            if outer_product_sampling:
                metrics[split]['discrete_action_logits'] = []
        metrics[split]['trajectory_length'] = []
        metrics[split]['frames'] = []
        metrics[split]['scene_name_or_num'] = []
        for i in range(episodes):
            with torch.no_grad():
                if scene_numbers is not None:
                    # If scene_numbers is provided, set up those scenes exactly
                    # the same way every time
                    scene_num = scene_numbers
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
                        use_gt_segmentation=use_gt_segmentation,
                        fusion_model=fusion_model,
                        outer_product_sampling=outer_product_sampling,
                        inverse_score=inverse_score,
                        zero_null_superpixel_features=zero_null_superpixel_features,
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
                if outer_product_sampling:
                    metrics[split]['discrete_action_logits'].append(
                            [logits.detach().cpu() for mask_scores in
                                trajectory_results['discrete_action_logits']])
            metrics[split]['trajectory_length'].append(
                    float(len(trajectory_results['frames'])))
            metrics[split]['frames'].append(trajectory_results['frames'])
            metrics[split]['scene_name_or_num'].append(
                    trajectory_results['scene_name_or_num'])

    model.train()
    #model.visual_model.resnet_model.model.eval()
    return metrics

def write_results(writer, results, train_steps, train_frames=None,
        train_trajectories=None, fusion_model='SuperpixelFusion',
        outer_product_sampling=False, single_interact=False, save_path=None):
    """
    Write results to SummaryWriter.
    """
    for split in results.keys():
        for metric, values in results[split].items():
            steps_name = 'steps/' + split + '/' + metric
            frames_name = 'frames/' + split + '/' + metric
            trajectories_name = 'trajectories/' + split + '/' + metric

            # TODO: these special cases need to be helper functions
            if metric in ['frames', 'scene_name_or_num']:
                continue
            if fusion_model == 'SuperpixelFusion' and (
                    ('all_action_scores' in metric and not
                        outer_product_sampling) or
                    ('discrete_action_logits' in metric and
                        outer_product_sampling)):
                # all_action_scores is a list of lists (for each trajectory) of
                # tensors (for each step)
                trajectory_flat_action_scores = []
                for trajectory_action_scores in values:
                    trajectory_flat_action_scores.extend(
                            trajectory_action_scores)
                # Does leaving the action scores as (num_trajectories *
                # steps_per_trajectory, num_actions) actually change anything?
                writer.add_histogram(steps_name,
                        torch.stack(trajectory_flat_action_scores),
                        train_steps)
                if train_frames is not None:
                    writer.add_histogram(frames_name,
                            torch.stack(trajectory_flat_action_scores),
                            train_frames)
                if train_trajectories is not None:
                    writer.add_histogram(trajectories_name,
                            torch.stack(trajectory_flat_action_scores),
                            train_trajectories)

                # Add per-action score histograms
                for action_i in range(len(values[0][0])):
                    action_name = (constants.SIMPLE_ACTIONS[action_i] if
                            single_interact else
                            constants.COMPLEX_ACTIONS[action_i])

                    flat_action_i_scores = []
                    for trajectory in range(len(values)):
                        flat_action_i_scores.extend([action_scores[action_i]
                            for action_scores in values[trajectory]])

                    writer.add_histogram(steps_name + '_' + action_name,
                            torch.stack(flat_action_i_scores), train_steps)
                    if train_frames is not None:
                        writer.add_histogram(frames_name + '_' + action_name,
                                flat_action_i_scores, train_frames)
                    if train_trajectories is not None:
                        writer.add_histogram(trajectories_name + '_' +
                                action_name, flat_action_i_scores,
                                train_trajectories)

                # Also histogram the top action (not necessarily the chosen
                # action, if sampling)
                if ('all_action_scores' in metric and not
                        outer_product_sampling):
                    action_argmaxes = [torch.argmax(action_scores).item() for
                            action_scores in trajectory_flat_action_scores]
                    writer.add_histogram('steps/' + split + '/' +
                            'action_argmaxes', action_argmaxes, train_steps,
                            bins='fd')
                    if train_frames is not None:
                        writer.add_histogram('frames/' + split + '/' +
                                'action_argmaxes', action_argmaxes,
                                train_frames, bins='fd')
                    if train_trajectories is not None:
                        writer.add_histogram('trajectories/' + split + '/' +
                                'action_argmaxes', action_argmaxes,
                                train_trajectories, bins='fd')
            elif ('all_action_scores' in metric and (fusion_model ==
                    'SuperpixelActionConcat' or outer_product_sampling)):
                # all_action_scores is a list of lists (for each trajectory) of
                # tensors (for each step). Flatten all scores, since each step
                # can have different numbers of action+superpixel scores, so
                # each trajectory can have different numbers of
                # action+superpixel scores if flattened by trajectory
                flat_action_scores = []
                actions = (constants.SIMPLE_ACTIONS if single_interact else
                        constants.COMPLEX_ACTIONS)
                per_action_scores = [[] for _ in actions]
                for trajectory_action_scores in values:
                    for action_scores in trajectory_action_scores:
                        flat_action_scores.extend(action_scores)
                        for action_i in range(len(constants.NAV_ACTIONS)):
                            per_action_scores[action_i].append(
                                    action_scores[action_i])
                        num_superpixels = \
                                superpixelactionconcat_get_num_superpixels(
                                        len(action_scores),
                                        single_interact=single_interact)
                        num_interact_actions = (1 if single_interact else
                                len(constants.INT_ACTIONS))
                        for action_i in range(num_interact_actions):
                            start = int(len(constants.NAV_ACTIONS) + action_i *
                                    num_superpixels)
                            end = int(start + num_superpixels)
                            per_action_scores[len(constants.NAV_ACTIONS) +
                                    action_i].extend(action_scores[start:end])

                writer.add_histogram(steps_name,
                        torch.stack(flat_action_scores), train_steps)
                if train_frames is not None:
                    writer.add_histogram(frames_name,
                            torch.stack(flat_action_scores), train_frames)
                if train_trajectories is not None:
                    writer.add_histogram(trajectories_name,
                            torch.stack(flat_action_scores),
                            train_trajectories)

                # Add per-action score histograms
                for action_i in range(len(actions)):
                    writer.add_histogram(steps_name + '_' + actions[action_i],
                            torch.stack(per_action_scores[action_i]),
                            train_steps)
                    if train_frames is not None:
                        writer.add_histogram(frames_name + '_' +
                                actions[action_i], per_action_scores[action_i],
                                train_frames)
                    if train_trajectories is not None:
                        writer.add_histogram(trajectories_name + '_' +
                                actions[action_i], per_action_scores[action_i],
                                train_trajectories)
            elif 'all_mask_scores' in metric:
                flat_mask_scores = []
                for trajectory_mask_scores in values:
                    for mask_scores in trajectory_mask_scores:
                        flat_mask_scores.extend(mask_scores)
                writer.add_histogram(steps_name,
                        torch.stack(flat_mask_scores), train_steps)
                if train_frames is not None:
                    writer.add_histogram(frames_name,
                            torch.stack(flat_mask_scores), train_frames)
                if train_trajectories is not None:
                    writer.add_histogram(trajectories_name,
                            torch.stack(flat_mask_scores),
                            train_trajectories)
            elif 'values' in metric:
                # Values is a list of lists (for each trajectory) of scalars
                flat_value_scores = []
                for value_scores in values:
                    flat_value_scores.extend(value_scores)
                writer.add_histogram(steps_name,
                        torch.stack(flat_value_scores),
                        train_steps)
                if train_frames is not None:
                    writer.add_histogram(frames_name,
                            torch.stack(flat_value_scores), train_frames)
                if train_trajectories is not None:
                    writer.add_histogram(trajectories_name,
                            torch.stack(flat_value_scores), train_trajectories)

                # Add per-step state-value histograms
                # Assumes that trajectories are all the same (fixed) length
                '''
                for i in range(len(values[0])):
                    writer.add_histogram(steps_name + '_' + str(i),
                            torch.stack([value_scores[i] for value_scores in
                                values]), train_steps)
                    writer.add_histogram(frames_name + '_' + str(i),
                            torch.stack([value_scores[i] for value_scores in
                                values]), train_frames)
                    if train_trajectories is not None:
                        writer.add_histogram(trajectories_name + '_' + str(i),
                                torch.stack([value_scores[i] for value_scores in
                                    values]), train_trajectories)
                avg_value = torch.mean(torch.stack(flat_value_scores)).item()
                writer.add_scalar('steps/' + split + '/' +
                        'avg_value', avg_value, train_steps)
                if train_frames is not None:
                    writer.add_scalar('frames/' + split + '/' + 'avg_value',
                            avg_value, train_frames)
                if train_trajectories is not None:
                    writer.add_scalar('trajectories/' + split + '/' +
                            'avg_value', avg_value, train_trajectories)
                '''
            elif 'pred_action_indexes' in metric:
                # pred_action_indexes is only used to compute (per action)
                # navigation and interaction action success rates
                continue
            elif 'action_successes' in metric:
                action_successes = defaultdict(list)
                actions = (constants.SIMPLE_ACTIONS if single_interact else
                        constants.COMPLEX_ACTIONS)
                # Hack to deal with the case when graphing averages over an
                # eval_interval
                if 'avg' in metric:
                    pred_action_indexes = results[split]['avg/pred_action_indexes']
                    all_action_scores = results[split]['avg/all_action_scores']
                else:
                    pred_action_indexes = results[split]['pred_action_indexes']
                    all_action_scores = results[split]['all_action_scores']
                for (trajectory_pred_action_indexes,
                        trajectory_action_successes,
                        trajectory_action_scores) in zip(
                                pred_action_indexes,
                                values,
                                all_action_scores):
                    for (pred_action_index,
                            action_success,
                            action_scores) in zip(
                                trajectory_pred_action_indexes,
                                trajectory_action_successes,
                                trajectory_action_scores):
                        if (fusion_model == 'SuperpixelFusion' and not
                                outer_product_sampling):
                            action_successes[actions[pred_action_index]] \
                                    .append(action_success)
                        elif fusion_model == 'SuperpixelActionConcat' or (
                                fusion_model == 'SuperpixelFusion' and
                                outer_product_sampling):
                           action_successes[
                                   superpixelactionconcat_index_to_action(
                                       pred_action_index, len(action_scores),
                                       single_interact=single_interact)] \
                                               .append(action_success)

                # Graph action success rate by action, by navigation or
                # interaction, and in total
                actions = (constants.SIMPLE_ACTIONS if single_interact else
                        constants.COMPLEX_ACTIONS)
                # We could compute the F1, but per action and for navigation
                # and interaction separately is enough, since navigation
                # actions aren't "the same weight" as interaction actions
                nav_successes = []
                int_successes = []
                for action in actions:
                    if len(action_successes[action]) == 0:
                        continue
                    action_avg_success = np.mean(action_successes[action])
                    writer.add_scalar(steps_name + '_' + action,
                            action_avg_success, train_steps)
                    if train_frames is not None:
                        writer.add_scalar(frames_name + '_' + action,
                                action_avg_success, train_frames)
                    if train_trajectories is not None:
                        writer.add_scalar(trajectories_name + '_' + action,
                                action_avg_success, train_trajectories)
                    if action in constants.NAV_ACTIONS:
                        nav_successes.extend(action_successes[action])
                    else:
                        int_successes.extend(action_successes[action])

                # Graph success rate by navigation or interaction
                if len(nav_successes) > 0:
                    nav_avg_success = np.mean(nav_successes)
                    writer.add_scalar(steps_name + '_navigation',
                            nav_avg_success, train_steps)
                    if train_frames is not None:
                        writer.add_scalar(frames_name + '_navigation',
                                nav_avg_success, train_frames)
                    if train_trajectories is not None:
                        writer.add_scalar(trajectories_name + '_navigation',
                                nav_avg_success, train_trajectories)
                if len(int_successes) > 0:
                    int_avg_success = np.mean(int_successes)
                    writer.add_scalar(steps_name + '_interaction',
                            int_avg_success, train_steps)
                    if train_frames is not None:
                        writer.add_scalar(frames_name + '_interaction',
                                int_avg_success, train_frames)
                    if train_trajectories is not None:
                        writer.add_scalar(trajectories_name + '_interaction',
                                int_avg_success, train_trajectories)

                # Graph overall success rate
                avg_success = np.mean(nav_successes + int_successes)
                writer.add_scalar(steps_name, avg_success, train_steps)
                if train_frames is not None:
                    writer.add_scalar(frames_name, avg_success, train_frames)
                if train_trajectories is not None:
                    writer.add_scalar(trajectories_name, avg_success,
                            train_trajectories)
            else:
                mean = np.mean(values)
                writer.add_scalar(steps_name, mean, train_steps)
                if train_frames is not None:
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

