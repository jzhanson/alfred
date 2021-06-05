import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import numpy as np
import torch

import gen.constants as constants

def delete_keys_from_dict(dict_del, lst_keys):
    """
    Delete the keys present in lst_keys from the dictionary.
    Loops recursively over nested dictionaries.
    """
    dict_foo = dict_del.copy()  #Used as iterator to avoid the 'DictionaryHasChanged' error
    for field in dict_foo.keys():
        if field in lst_keys:
            del dict_del[field]
        if type(dict_foo[field]) == dict:
            delete_keys_from_dict(dict_del[field], lst_keys)
    return dict_del


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_partial_model(pretrained_dict, model):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def stack_frames(frames, frame_stack=1, zero_fill_frame_stack=False,
        device=torch.device('cpu')):
    """
    stack_frames takes a list of tensors, one tensor per trajectory, so if only
    one trajectory is being stacked, may need to wrap in an outer list and
    unwrap afterwards.

    stack_frames returns a list of tensors, one tensor per trajectory, and
    there are the same number of stacked frames as there were original frames,
    so if you only want one stacked frame, pass frame_stack frames and only
    take the last one.
    """
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

def superpixelactionconcat_get_num_superpixels(num_scores,
        single_interact=False, navigation_superpixels=False):
    if navigation_superpixels:
        return num_scores / (len(constants.SIMPLE_ACTIONS) if single_interact
                else len(constants.COMPLEX_ACTIONS))
    else:
        return (num_scores - len(constants.NAV_ACTIONS)) / (1 if
                single_interact else len(constants.INT_ACTIONS))

def superpixelactionconcat_index_to_action(index, num_scores,
        single_interact=False, navigation_superpixels=False):
    if navigation_superpixels:
        actions = (constants.SIMPLE_ACTIONS if single_interact else
                constants.COMPLEX_ACTIONS)
        num_superpixels = superpixelactionconcat_get_num_superpixels(
                num_scores, single_interact=single_interact,
                navigation_superpixels=navigation_superpixels)
        return actions[int(index // num_superpixels)]
    else:
        if index < len(constants.NAV_ACTIONS):
            return constants.NAV_ACTIONS[index]
        int_actions = ([constants.ACTIONS_INTERACT] if single_interact else
                constants.INT_ACTIONS)
        num_superpixels = superpixelactionconcat_get_num_superpixels(
                num_scores, single_interact=single_interact)
        return int_actions[int((index - len(constants.NAV_ACTIONS)) //
            num_superpixels)]

# From https://github.com/pytorch/pytorch/issues/7014#issuecomment-388931028
# Safe multinomial sampling, even if there are infinite values in probs or
# logits
def multinomial(probs=None, logits=None, temperature=1, num_samples=1,
                     min_prob=1e-20, max_logit=1e+20,
                     min_temperature=1e-20, max_temperature=1e+20):
    if probs is not None:
        probs = probs.clamp(min=min_prob)
        logits = torch.log(probs)
    logits = logits.clamp(max=max_logit)
    temperature = np.clip(temperature, min_temperature, max_temperature)
    logits = (logits - logits.max()) / temperature
    probs = torch.exp(logits)
    return torch.multinomial(probs, num_samples)

# Loosely from https://github.com/dgriff777/rl_a3c_pytorch/blob/master/utils.py
def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
            shared_model.parameters()):
        if not param.requires_grad:
            continue
        elif param.grad is None:
            print('param', name, 'requires grad but grad is None')
        # Moved checking if condition that checks if model is the same as
        # shared_model outside fo this function
        if not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()
