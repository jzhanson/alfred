import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import time

import gen.constants as constants
from models.model.args import parse_args
from models.nn.resnet import Resnet
from models.nn.ie import ResnetWrapper
from models.model.rl_interaction import load_checkpoint, load_optimizer
from models.model.visual_probe import train
from models.utils.shared_optim import SharedRMSprop, SharedAdam
from data.interaction_dataset import InteractionDataset

def load_interaction_checkpoint(load_path, model):
    checkpoint = torch.load(load_path)
    renamed_checkpoint_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if 'visual_model' in k:
            renamed_k = k.replace('visual_model', 'resnet_model')
            renamed_checkpoint_state_dict[renamed_k] = v
    model.load_state_dict(renamed_checkpoint_state_dict, strict=False)

def setup_dataloaders(args):
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset_type == 'imagenet':
        train_dataset = datasets.ImageNet(args.dataset_path, split='train',
                download=False, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.dataloader_workers,
                pin_memory=True)
        eval_dataset = datasets.ImageNet(args.dataset_path, split='val',
                download=False, transform=transform)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.dataloader_workers,
                pin_memory=True)
    elif 'interaction' in args.dataset_type:
        # TODO: come up with a different eval dataset?
        dataset = InteractionDataset(args.dataset_path,
                dataset_type=args.dataset_type.split('_')[1],
                max_trajectory_length=args.max_trajectory_length,
                high_res_images=args.high_res_images,
                scene_target_type=args.interaction_scene_target_type,
                scene_binary_labels=args.interaction_scene_binary_labels,
                excluded_object_types=args.excluded_object_types,
                object_distance_threshold=args.object_distance_threshold)
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.dataloader_workers,
                pin_memory=True)
        eval_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.dataloader_workers,
                pin_memory=True)
    return train_dataloader, eval_dataloader

# Need to take gpu_id instead of device as argument because resnet needs gpu_id
def setup_model(args, gpu_id=None):
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    if gpu_id is not None:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    resnet_args = Namespace(visual_model='resnet', gpu=gpu_id is not None,
            gpu_index=gpu_id if gpu_id is not None else -1)
    if 'resnet' in args.visual_model:
        visual_model = Resnet(resnet_args, use_conv_feat=False,
                pretrained=args.pretrained_visual_model,
                frozen=args.frozen_visual_model,
                layers=args.custom_resnet_layers,
                inplanes=args.custom_resnet_inplanes,
                planes=args.custom_resnet_planes)
    else:
        print("visual model '" + args.visual_model + "' not supported")

    if args.visual_fc_units is None:
        # This case probably won't happen since we want to do classification
        # with the model output
        args.visual_fc_units = []
    elif type(args.visual_fc_units) is int:
        args.visual_fc_units = [args.visual_fc_units]
    visual_model = ResnetWrapper(resnet_model=visual_model,
            fc_units=args.visual_fc_units, dropout=args.dropout,
            use_tanh=args.use_tanh)

    try:
        visual_model = visual_model.to(device)
    except:
        visual_model = visual_model.to(device)

    return visual_model

def setup_optimizer(model, optimizer_name='', lr=0.01, shared=False):
    parameters = model.parameters()
    if shared:
        # SharedSGD not implemented
        if optimizer_name == 'rmsprop':
            optimizer = SharedRMSprop(parameters, lr=lr)
        elif 'adam' in optimizer_name:
            amsgrad = 'amsgrad' in optimizer_name
            optimizer = SharedAdam(parameters, lr=lr, amsgrad=amsgrad)
        optimizer.share_memory()
    else:
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, lr=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=lr)
        elif 'adam' in optimizer_name:
            amsgrad = 'amsgrad' in optimizer_name
            optimizer = optim.Adam(parameters, lr=lr, amsgrad=amsgrad)
    return optimizer

def setup_train(rank, args, shared_model, shared_optimizer, train_steps_sync):
    # Set random seed for worker process
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if args.gpu_ids is not None:
        torch.cuda.manual_seed(args.seed + rank)

    if args.gpu_ids is not None:
        gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
        device = torch.device('cuda:' + str(gpu_id))
    else:
        gpu_id = None
        device = torch.device('cpu')

    train_dataloader, eval_dataloader = setup_dataloaders(args)

    # Here, we allow the model to be the same as the shared_model in case
    # we're only running a single worker so we don't have two copies of
    # models. I guess this could also be used for multiple workers with one
    # worker process's model being the shared model, but that kind of
    # defeats the purpose of having a shared model in the first place
    # (which is to allow the possibility of locking, and self-locking is
    # more complex and less clean)
    if args.num_processes > 1:
        model = setup_model(args, gpu_id=gpu_id)
        # Load checkpoint but not optimizer
        if args.load_path is not None:
            if args.loading_interaction:
                load_interaction_checkpoint(args.load_path, model)
            else:
                load_checkpoint(args.load_path, model, None, None)
    else:
        # Shared models will already be on the single GPU that is being used.
        # Checkpoint is already loaded
        model = shared_model

    if shared_optimizer is None:
        optimizer = setup_optimizer(shared_model,
                optimizer_name=args.optimizer,
                lr=args.lr, shared=False)
        # Only load optimizers if not loading an interaction checkpoint
        if args.load_path is not None and not args.loading_interaction:
            # Do some filename tomfoolery to see if there are saved optimizer
            # files for the current worker process. If not (i.e. running with
            # more processes than before), load checkpoint's saved optimizer
            # state
            load_dir, checkpoint_name = os.path.split(args.load_path)
            if checkpoint_name == 'model.pth': # Wasn't using save_intermediate
                optimizer_checkpoint_name = 'optimizer_' + str(rank) + '.pth'
            else:
                # Choose the optimizer closest to the checkpoint for that worker
                load_step = int(checkpoint_name.split('.')[0].split('_')[1])
                optimizer_steps = [int(fname.split('.')[0].split('_')[1]) for
                        fname in os.listdir(load_dir) if 'optimizer' in fname
                        and int(fname.split('.')[0].split('_')[2]) == rank]
                if len(optimizer_steps) > 0:
                    closest_optimizer_step = min(optimizer_steps, key=lambda
                            x:abs(x - load_step))
                else:
                    closest_optimizer_step = -1
                optimizer_checkpoint_name = ('optimizer_' +
                        str(closest_optimizer_step) + '_' + str(rank) + '.pth')

            optimizer_load_path = os.path.join(load_dir,
                    optimizer_checkpoint_name)
            if os.path.isfile(optimizer_load_path):
                load_optimizer(optimizer_load_path, optimizer)
            else:
                load_optimizer(args.load_path, optimizer)
    else:
        # Likewise, shared optimizer state is already loaded
        optimizer = shared_optimizer

    train(rank, args.num_processes, model, shared_model, train_dataloader,
            eval_dataloader, optimizer, train_steps_sync,
            batch_size=args.batch_size, sync_on_epoch=args.sync_on_epoch,
            dataset_type=args.dataset_type,
            interaction_scene_binary_labels=
            args.interaction_scene_binary_labels,
            max_grad_norm=args.max_grad_norm, eval_interval=args.eval_interval,
            max_steps=args.max_steps, device=device, save_path=args.save_path,
            save_intermediate=args.save_intermediate)

if __name__ == '__main__':
    args = parse_args()

    # Set random seed for everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_ids is not None:
        torch.cuda.manual_seed(args.seed)

    if args.save_path is not None and not os.path.isdir(args.save_path):
        print('making directory', args.save_path)
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'args.json'), 'w') as jsonfile:
        json.dump(args.__dict__, jsonfile)

    if args.gpu_ids is not None and type(args.gpu_ids) is int:
        args.gpu_ids = [args.gpu_ids]
    if type(args.excluded_object_types) is str:
        args.excluded_object_types = [args.excluded_object_types]

    # Put shared models on GPU if there's only one process and we're using GPU
    # to save the CPU memory of a "useless" shared_model
    #
    # Be careful about moving models carelessly between devices because the
    # custom Resnet class has a self.device member that won't be changed by
    # model.to(device)!
    shared_model = setup_model(args, gpu_id=args.gpu_ids[0] if args.gpu_ids is
            not None and args.num_processes == 1 else None)

    if args.shared_optimizer:
        shared_optimizer = setup_optimizer(shared_model,
                optimizer_name=args.optimizer, lr=args.lr, shared=True)
    else:
        shared_optimizer = None

    if args.load_path is not None:
        if args.loading_interaction:
            train_steps = 0
            load_interaction_checkpoint(args.load_path, shared_model)
        else:
            train_steps = load_checkpoint(args.load_path, shared_model,
                    None, shared_optimizer)
    else:
        train_steps = 0

    print('shared model parameters: ' + str(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad)))
    print('total parameters: ' + str(sum(p.numel() for p in
        shared_model.parameters() if p.requires_grad) * args.num_processes))

    # The rest of this file is from
    # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/main.py, which is
    # from https://github.com/pytorch/examples/tree/master/mnist_hogwild
    #
    # Don't know why dgriff777 only sets start method if running with gpu, code
    # hangs when running on CPU unless I set it
    mp.set_start_method('spawn')

    processes = []

    # Signed int should be large enough :P
    train_steps_sync = mp.Value('i', train_steps)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=setup_train, args=(rank, args, shared_model,
            shared_optimizer, train_steps_sync))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

