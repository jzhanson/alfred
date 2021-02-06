import os
import sys
import multiprocessing as mp
import threading
from collections import defaultdict
import json
import argparse
import time

import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'data'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
import constants

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import cv2
from env.thor_env import ThorEnv
from env.find_one import FindOne


parser = argparse.ArgumentParser()
#parser.add_argument('--data_path', type=str, default="data/2.1.0")
parser.add_argument('--num-threads', type=int, default=1)


with open('/home/jzhanson/alfred/data/find_one_find_pngs/obj_type_to_index.json', 'r') as jsonfile:
    obj_type_to_index = json.load(jsonfile)

with open('/home/jzhanson/alfred/data/find_one_find_pngs/train.json', 'r') as jsonfile:
    train = json.load(jsonfile)
with open('/home/jzhanson/alfred/data/find_one_find_pngs/valid_seen.json', 'r') as jsonfile:
    valid_seen = json.load(jsonfile)
with open('/home/jzhanson/alfred/data/find_one_find_pngs/valid_unseen.json', 'r') as jsonfile:
    valid_unseen = json.load(jsonfile)

global current_trajectory_index
current_trajectory_index = 0
pbar = tqdm(total=len(train))

lock = threading.Lock() # To protect current_trajectory_index

# TODO: do this for valid_seen and valid_unseen
def run(thread_id):
    thor_env = ThorEnv()
    fo = FindOne(thor_env, obj_type_to_index)

    # trajectory index, target type, receptacle type, openable, open
    targets_receptacles = []
    skipped_trajectory_indexes = []

    global current_trajectory_index
    while True:
        lock.acquire()
        if current_trajectory_index < len(train):
            trajectory_index = current_trajectory_index
            current_trajectory_index += 1
            '''
            print('Remaining trajectories: ' + str(len(train) -
                current_trajectory_index))
            '''
            pbar.update(1)
            if current_trajectory_index == len(train):
                pbar.close()
            lock.release()
        else:
            lock.release()
            break

        trajectory = train[trajectory_index]

        # Load traj_data and see if any target is inside any other objects
        traj_data_path = os.path.join(trajectory['path'], 'traj_data.json')
        with open(traj_data_path, 'r') as jsonfile:
            traj_data = json.load(jsonfile)
        if type(trajectory['high_idx']) is list:
            high_idx = trajectory['high_idx'][0]
        else:
            high_idx = trajectory['high_idx']
        try:
            fo.load_from_traj_data(traj_data, high_idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print ("Error: " + repr(e))
            print ("Skipping " + str(trajectory_index))
            skipped_trajectory_indexes.append(trajectory_index)
            continue

        # Unfortunately, because our saved trajectory does not contain the
        # objectId but only the object type, we have to iterate through all
        # objects in the scene
        target_type = constants.OBJECTS_LOWER_TO_UPPER[trajectory['target']]
        for obj in fo.env.last_event.metadata['objects']:
            # There may be multiple objects with the target type in
            # a scene, but we'll just track all of them
            if target_type == obj['objectType'] and obj['parentReceptacles'] \
                    is not None:
                for parent_receptacle_id in obj['parentReceptacles']:
                    parent_receptacle = fo.env.last_event.get_object(
                        parent_receptacle_id)
                    if parent_receptacle['openable']:
                        targets_receptacles.append((trajectory_index,
                            target_type, parent_receptacle['objectType'],
                            parent_receptacle['isOpen']))

    # Dump to JSON to sort later
    with open('/home/jzhanson/alfred/data/train_target_occurrence_stats_' +
            str(thread_id) + '.json', 'w+') as outfile:
        json.dump(targets_receptacles, outfile, indent=4)
    with open('/home/jzhanson/alfred/data/train_target_occurrence_skipped_' +
            str(thread_id) + '.json', 'w+') as outfile:
        json.dump(skipped_trajectory_indexes, outfile, indent=4)
    print('Skipped trajectory indexes:')
    print(skipped_trajectory_indexes)

    thor_env.stop()

#if __name__ == '__main__':
args = parser.parse_args()


threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run, args=(n,))
    threads.append(thread)
    thread.start()
    time.sleep(1)

'''
counts = defaultdict(int)j

with open('/home/jzhanson/alfred/data/find_one_find_pngs/train.json', 'r') as jsonfile:
    train = json.load(jsonfile)

for trajectory in tqdm(train):
    counts[trajectory['target']] += 1

sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)

print('total trajectories: ' + str(len(train)))

print('target counts: ')
print(sorted_counts)
'''

'''
# Load scenes and figure out which objects are impossible targets
# (hidden in things) from https://arxiv.org/pdf/1812.00971.pdf
# Can manually remove objects like Bed or CounterTop that are "too large"
# scene number: [(object id, parent receptacle id, parent receptacle is open)]
contained_objects = defaultdict(list)
for scene_num in tqdm(constants.SCENE_NUMBERS):
    thor_env.reset(scene_num)
    for obj in thor_env.last_event.metadata['objects']:
        if obj['parentReceptacles'] is not None:
            for parent_receptacle_id in obj['parentReceptacles']:
                parent_receptacle = thor_env.last_event.get_object(
                        parent_receptacle_id)
                if parent_receptacle['openable']:
                    contained_objects[scene_num].append((obj['objectId'],
                        parent_receptacle_id, parent_receptacle['isOpen']))

with open('/home/jzhanson/alfred/data/scenes_contained_objects.json', 'w') \
        as outfile:
    json.dump(contained_objects, outfile, indent=4)

# Lightly analyze the objects and containers
with open('/home/jzhanson/alfred/data/scenes_contained_objects.json', 'r') \
        as jsonfile:
    contained_objects = json.load(jsonfile)

flat_contained_objects = []
for scene_number, tuples_list in contained_objects.items():
    for (object_id, parent_receptacle_id, parent_receptacle_open) in tuples_list:
        flat_contained_objects.append((scene_number, object_id,
            parent_receptacle_id, parent_receptacle_open))

df = pd.DataFrame(data=flat_contained_objects, columns=['scene_num',
    'object_id', 'parent_id', 'parent_open'])

print('total contained occurrences: ' + str(len(df)))
print('total closed occurrences: ' + str(len(df[df['parent_open'] == False])))

containers = defaultdict(int)
objects = defaultdict(int)
for _, object_id, parent_id, _ in flat_contained_objects:
    containers[parent_id.split('|')[0]] += 1
    objects[object_id.split('|')[0]] += 1

# Toilet can be open/closed but objects can only be put on top of the toilet
print('enclosed objects: ')
print(sorted(objects.items(), key=lambda x: x[1], reverse=True))

print('containers: ')
print(sorted(containers.items(), key=lambda x: x[1], reverse=True))
'''
