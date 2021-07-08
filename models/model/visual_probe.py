import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import random
from collections import defaultdict
import time
from time import sleep

import numpy as np
import torch
import torch.nn.functional as F

import gen.constants as constants
from models.utils.helper_utils import stack_frames, ensure_shared_grads
from models.model.rl_interaction import (save_checkpoint, save_optimizer,
        load_checkpoint, load_optimizer, write_results)
import cv2

from tensorboardX import SummaryWriter

def evaluate(model, shared_model, eval_dataloader, dataset_type='imagenet',
        interaction_scene_binary_labels=True, device=torch.device('cpu')):
    # First, load state dict
    if model != shared_model:
        model.load_state_dict(shared_model.state_dict())
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_dataloader:
            output = model(data.cpu())
            if (dataset_type == 'imagenet' or dataset_type ==
                    'interaction_object'):
                output_log_probs = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output_log_probs, target.to(device))
                max_values, max_indexes = torch.max(output_log_probs, 1)
                correct += max_indexes.eq(target.to(device)).sum().item()
            elif dataset_type == 'interaction_scene':
                if interaction_scene_binary_labels:
                    loss = F.binary_cross_entropy_with_logits(output,
                            target.to(device))
                else:
                    loss = F.mse_loss(output, target.to(device))
            eval_loss += loss
    model.train()
    if dataset_type == 'imagenet' or dataset_type == 'interaction_object':
        accuracy = correct / len(eval_dataloader.dataset)
    elif dataset_type == 'interaction_scene':
        accuracy = None
    return eval_loss.item(), accuracy

def take_step(model, shared_model, optimizer, data, target,
        dataset_type='imagenet', interaction_scene_binary_labels=True,
        max_grad_norm=50, device=torch.device('cpu')):
    output = model(data.cpu())
    if dataset_type == 'imagenet' or dataset_type == 'interaction_object':
        output_log_probs = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output_log_probs, target.to(device))
    elif dataset_type == 'interaction_scene':
        if interaction_scene_binary_labels:
            loss = F.binary_cross_entropy_with_logits(output,
                    target.to(device))
        else:
            loss = F.mse_loss(output, target.to(device))
    optimizer.zero_grad()
    try:
        loss.backward(retain_graph=True)
    except:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if model != shared_model:
        ensure_shared_grads(model, shared_model, gpu=(not device ==
            torch.device('cpu')))
    optimizer.step()
    if dataset_type == 'imagenet' or dataset_type == 'interaction_object':
        max_values, max_indexes = torch.max(output_log_probs, 1)
        correct = max_indexes.eq(target.to(device)).sum().item()
        accuracy = correct / data.shape[0]
    elif dataset_type == 'interaction_scene':
        accuracy = None
    return loss, accuracy

# A lot of this function is taken from train() in
# models/model/rl_interaction.py
# TODO: consider sharing multiprocessing and eval logic with a shared train
# function and separate metrics and batch+step functions
def train(rank, num_processes, model, shared_model, train_dataloader,
        eval_dataloader, optimizer, train_steps_sync, batch_size=50,
        sync_on_epoch=False, dataset_type='imagenet',
        interaction_scene_binary_labels=False, max_grad_norm=50,
        eval_interval=10, max_steps=1000000, device=torch.device('cpu'),
        save_path=None, save_intermediate=False):
    # If multiple processes try to write to different SummaryWriters, only one
    # tensorboard logfile is generated but the logged data seems to not be
    # understandable by tensorboard. Also, GlobalSummaryWriter's _writer global
    # doesn't seem to be shared across processes, even if initialized in the
    # parent process
    # TODO: log from all processes, probably using wandb. Probably will need to
    # pass rank and num_procs arguments to this function.
    if save_path is not None:
        writer = SummaryWriter(log_dir=os.path.join(save_path,
            'tensorboard_logs'))
    else:
        writer = None

    # If loading from file, metrics will be blank, but that's okay because
    # train_steps will be accurate, so it will just pick up where it left off
    last_metrics = {}
    last_metrics['loss'] = []
    last_metrics['accuracy'] = []

    start_time = time.time()
    last_eval_time = start_time
    # For checking if train_steps_sync has passed an eval_interval
    last_train_steps_local = None
    train_steps_local = None
    model.train()
    if not sync_on_epoch:
        train_dataloader_iterator = iter(train_dataloader)
    while True:
        # "Grab ticket" and increment train_steps_sync with the intention of
        # rolling out that trajectory and taking that gradient step We have to
        # increment here in case we need to save trajectory_info from all
        # threads
        with train_steps_sync.get_lock():
            if train_steps_local is None: # First iteration, even if loading
                train_steps_local = train_steps_sync.value
                last_train_steps_local = train_steps_local
            else:
                last_train_steps_local = train_steps_local
                train_steps_local = train_steps_sync.value
            train_steps_sync.value += 1
        if train_steps_local >= max_steps:
            break

        # Load state dict from shared model
        if model != shared_model:
            # We don't load optimizers for each worker process - if having a
            # shared/synchronized optimizer state is needed then there's a
            # shared_optimizer option in models/train/train_rl_ie.py
            model.load_state_dict(shared_model.state_dict())

        # TODO: add option for locking update
        if sync_on_epoch:
            for data, target in train_dataloader:
                # accuracy is None if task is not classification
                loss, accuracy = take_step(model, shared_model, optimizer,
                        data, target, dataset_type=dataset_type,
                        interaction_scene_binary_labels=
                        interaction_scene_binary_labels,
                        max_grad_norm=max_grad_norm, device=device)
                last_metrics['loss'].append(loss.item())
                if (dataset_type == 'imagenet' or dataset_type ==
                        'interaction_object'):
                    last_metrics['accuracy'].append(accuracy)
        else:
            # Pattern from
            # https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
            # for turning a Dataloader into an iterator
            try:
                data, target = next(train_dataloader_iterator)
            except StopIteration:
                train_dataloader_iterator = iter(train_dataloader)
                data, target = next(train_dataloader_iterator)
            # accuracy is None if task is not classification
            loss, accuracy = take_step(model, shared_model, optimizer,
                    data, target, dataset_type=dataset_type,
                    interaction_scene_binary_labels=
                    interaction_scene_binary_labels,
                    max_grad_norm=max_grad_norm, device=device)
            last_metrics['loss'].append(loss.item())
            if (dataset_type == 'imagenet' or dataset_type ==
                    'interaction_object'):
                last_metrics['accuracy'].append(accuracy)

        if writer is not None and rank == 0:
            results = {}
            results['train'] = {}
            for metric in last_metrics.keys():
                if len(last_metrics[metric]) == 0:
                    continue
                results['train'][metric] = [last_metrics[metric][-1]]
            # Don't write training results to file
            write_results(writer, results, train_steps_local, save_path=None)

        # Save checkpoint every N trajectories, collect/print stats
        # If an eval_interval was passed, execute code - hopefully
        # eval_interval is set to something large enough relative to
        # num_processes that multiple eval_intervals can't be passed in one go
        if (last_train_steps_local // eval_interval != train_steps_local //
                eval_interval):
            print('steps %d ' % train_steps_local)
            current_time = time.time()
            total_frames = train_steps_local * batch_size
            if sync_on_epoch:
                total_frames *= len(train_dataloader)
            total_fps = total_frames / (current_time - start_time)
            eval_interval_frames = eval_interval * batch_size
            if sync_on_epoch:
                eval_interval_frames *= len(train_dataloader)
            eval_interval_fps = eval_interval_frames / (current_time -
                    last_eval_time)
            print('total fps since start %.6f' %
                    total_fps)
            print('total fps over last %d steps %d frames %.6f' %
                    (eval_interval, eval_interval_frames, eval_interval_fps))
            process_eval_frames = len(last_metrics['loss']) * batch_size
            process_fps = process_eval_frames / (time.time()
                    - last_eval_time)
            print('rank %d fps over last %d steps %d frames %.6f' % (rank,
                len(last_metrics['loss']), process_eval_frames,
                process_fps))
            # Output FPS to tensorboard SummaryWriter
            if rank == 0:
                writer.add_scalar('steps/train/fps_total_since_start',
                        total_fps, train_steps_local)
                writer.add_scalar('steps/train/fps_total_last_interval',
                        eval_interval_fps, train_steps_local)
                writer.add_scalar('steps/train/fps_process_last_interval',
                        process_fps, train_steps_local)

            for metric, values in last_metrics.items():
                last_metrics[metric] = []

            if save_path is not None:
                # TODO: add locked saving
                # Due to the difficulties of inter-process communication, it's
                # hard to guarantee that the checkpoint and each optimizer
                # state is from the same moment in wall clock time, so the best
                # we do is save checkpoints/optimizer states in intervals based
                # on each process's observed values of train_steps_sync, which
                # roughly happen at the same time due to the eval_interval
                # interval wraparound logic
                if rank == 0:
                    save_checkpoint(shared_model, optimizer, train_steps_local,
                            save_path=save_path,
                            save_intermediate=save_intermediate)
                else:
                    # Multiple copies of a shared optimizer state will be saved
                    # here by each worker process, but they won't be loaded, if
                    # a shared optimizer is being used when loading, and it's a
                    # little simpler than passing a shared_optimizer argument
                    # to this function
                    save_optimizer(rank, optimizer, train_steps_local,
                            save_path=save_path,
                            save_intermediate=save_intermediate)
            if rank == 0:
                eval_loss, eval_accuracy = evaluate(model, shared_model,
                        eval_dataloader, dataset_type=dataset_type,
                        interaction_scene_binary_labels=
                        interaction_scene_binary_labels,
                        device=device)
                print('eval loss %.6f' % eval_loss)
                writer.add_scalar('steps/eval/loss',
                        eval_loss, train_steps_local)
                if (dataset_type == 'imagenet' or dataset_type ==
                        'interaction_object'):
                    print('eval accuracy %.6f' % eval_accuracy)
                    writer.add_scalar('steps/eval/accuracy',
                            eval_accuracy, train_steps_local)

            last_eval_time = time.time()

    if save_path is not None:
        print('steps %d ' % train_steps_local)
        if rank == 0:
            # Wait until all other processes have taken their last (extra)
            # ticket/increment before saving checkpoint to make sure that
            # max_steps gradient updates have happened
            while train_steps_local < max_steps + num_processes - 1:
                # Other solutions that aren't sleepwaiting are a lot more
                # complex for not much gain
                sleep(1)
                with train_steps_sync.get_lock():
                    train_steps_local = train_steps_sync.value
            save_checkpoint(shared_model, optimizer, max_steps,
                    save_path=save_path,
                    save_intermediate=save_intermediate)
            eval_loss, eval_accuracy = evaluate(model, shared_model,
                    eval_dataloader, dataset_type=dataset_type,
                    interaction_scene_binary_labels=
                    interaction_scene_binary_labels,
                    device=device)
            print('eval loss %.6f' % eval_loss)
            writer.add_scalar('steps/eval/loss',
                    eval_loss, train_steps_local)
            if (dataset_type == 'imagenet' or dataset_type ==
                    'interaction_object'):
                print('eval accuracy %.6f' % eval_accuracy)
                writer.add_scalar('steps/eval/accuracy',
                        eval_accuracy, train_steps_local)
        else:
            save_optimizer(rank, optimizer, max_steps,
                    save_path=save_path,
                    save_intermediate=save_intermediate)

