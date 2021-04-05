import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

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
        single_interact=False):
    return (num_scores - len(constants.NAV_ACTIONS)) / (1 if single_interact
            else len(constants.INT_ACTIONS))

def superpixelactionconcat_index_to_action(index, num_scores,
        single_interact=False):
    if index < len(constants.NAV_ACTIONS):
        return constants.NAV_ACTIONS[index]
    int_actions = ([constants.ACTIONS_INTERACT] if single_interact else
            constants.INT_ACTIONS)
    num_superpixels = superpixelactionconcat_get_num_superpixels(num_scores,
            single_interact=single_interact)
    return int_actions[int((index - len(constants.NAV_ACTIONS)) //
        num_superpixels)]
