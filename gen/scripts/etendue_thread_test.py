# Initially taken from https://gist.github.com/etendue/20c66b694b35568532651f5d7c00252c from https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a
# Also can look at Multi-threaded example at https://allenai.github.io/ai2thor-v2.1.0-documentation/examples
import multiprocessing as mp
#import ai2thor.controller
import numpy as np
import time
import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'env'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
import random
import cv2
from env.thor_env import ThorEnv
import gen.constants as constants
from tqdm import tqdm

'''
actions = [
    'MoveAhead',
    'MoveBack',
    'MoveRight',
    'MoveLeft',
    'LookUp',
    'LookDown',
    'RotateRight',
    'RotateLeft',
    'OpenObject',
    'CloseObject',
    'PickupObject'
    # 'PutObject'
    # Teleport and TeleportFull but these shouldn't be allowable actions for an agent
]
'''
actions = constants.NAV_ACTIONS

def worker(steps_per_proc, sync_event, queue, gpu_id=0, actions=actions,
        init_kwargs={}, reset_kwargs={}):

    thor_env = ThorEnv(**init_kwargs)
    print("Worker with pid:", os.getpid(), "is intialized")
    np.random.seed(os.getpid())
    thor_env.reset(constants.SCENE_NUMBERS[np.random.randint(
        len(constants.SCENE_NUMBERS))], **reset_kwargs)
    #inform main process that intialization is successful
    queue.put(1)
    sync_event.wait()
    for _ in range(steps_per_proc):
        a = np.random.randint(len(actions))
        event = thor_env.step(dict(action=actions[a]))
        #print(actions[a], event.metadata['lastActionSuccess'],
        #        event.metadata['errorMessage'])

    queue.put(1)
    print("Worker with pid:", os.getpid(), "finished job")
    sync_event.clear()
    sync_event.wait()
    thor_env.stop()

def single_worker(num_envs, steps_per_proc, actions=actions, init_kwargs={},
        reset_kwargs={}):

    thor_envs = []
    for i in range(num_envs):
        thor_env = ThorEnv(**init_kwargs)
        thor_env.reset(1, **reset_kwargs)
        thor_envs.append(thor_env)

    np.random.seed(42)
    start = time.time()
    for _ in tqdm(range(steps_per_proc)):

        thor_env.reset(1, **reset_kwargs)
        #thor_env.reset(np.random.choice(list(range(1, 31))), **reset_kwargs)
        int_actions = np.random.randint(len(actions),size=num_envs)
        for thor_env, a in zip(thor_envs, int_actions):
            event = thor_env.step(dict(action=actions[a]))
            #print(actions[a], event.metadata['lastActionSuccess'],
            #        event.metadata['errorMessage'])
    total_time = time.time() - start
    print("FPS:", steps_per_proc * num_envs / total_time)
    for thor_env in thor_envs:
        thor_env.stop()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-processes','-mp', action='store_true',
            default=False,
            help="if False, then only one process creates multi envs")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--num-envs','-n', type=int, default=1)
    # https://github.com/allenai/ai2thor/blob/main/ai2thor/_quality_settings.py
    parser.add_argument('--quality', type=str, default='MediumCloseFitShadows')
    parser.add_argument('--render-image', dest='render_image',
            action='store_true')
    parser.add_argument('--no-render-image', dest='render_image',
            action='store_false')
    parser.set_defaults(render_image=True)
    parser.add_argument('--render-depth-image', dest='render_depth_image',
            action='store_true')
    parser.add_argument('--no-render-depth-image', dest='render_depth_image',
            action='store_false')
    parser.set_defaults(render_depth_image=True)
    parser.add_argument('--render-class-image', dest='render_class_image',
            action='store_true')
    parser.add_argument('--no-render-class-image', dest='render_class_image',
            action='store_false')
    parser.set_defaults(render_class_image=True)
    parser.add_argument('--render-object-image', dest='render_object_image',
            action='store_true')
    parser.add_argument('--no-render-object-image', dest='render_object_image',
            action='store_false')
    parser.set_defaults(render_object_image=True)
    parser.add_argument('--visibility-distance', type=float, default=1.5)

    args = parser.parse_args()

    init_kwargs = {
            'quality' : args.quality,
            'x_display' : '2'}
    reset_kwargs = {
            'render_image' : args.render_image,
            'render_depth_image' : args.render_depth_image,
            'render_class_image' : args.render_class_image,
            'render_object_image' : args.render_object_image,
            'visibility_distance' : args.visibility_distance,
            }

    if args.multi_processes:
        num_proc = args.num_envs
        steps_per_proc = args.steps // num_proc
        gpuIds = [i % args.gpus for i in range(num_proc)]
        events = [mp.Event() for _ in range(num_proc)]
        for e in events:
            e.clear()
        queue = mp.SimpleQueue()
        processes = []
        for i in range(num_proc):
            p = mp.Process(target=worker, args=(steps_per_proc, events[i],
                queue, gpuIds[i], init_kwargs, reset_kwargs))
            p.start()
            processes.append(p)

        initialized_processes = 0
        while(initialized_processes < num_proc):
            queue.get()
            initialized_processes +=1

        start_time= time.time()
        # start test
        for e in events:
            e.set()

        finished_processes = 0
        while (finished_processes < num_proc):
            queue.get()
            finished_processes += 1

        total_time = time.time() - start_time

        print("FPS:", args.steps / total_time)
        # inform process to exit
        for e in events:
            e.set()
        for p in processes:
            p.join()
        print("Test finished")

    else:
        steps_per_proc = args.steps // args.num_envs
        single_worker(args.num_envs, steps_per_proc, init_kwargs=init_kwargs,
                reset_kwargs=reset_kwargs)
