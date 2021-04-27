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

from tqdm import tqdm

import cv2
from env.thor_env import ThorEnv
from env.find_one import FindOne


parser = argparse.ArgumentParser()
#parser.add_argument('--data_path', type=str, default="data/2.1.0")
parser.add_argument('--num-threads', type=int, default=1)


with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/find_one_find_pngs/obj_type_to_index.json'), 'r') as jsonfile:
    obj_type_to_index = json.load(jsonfile)

with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/find_one_find_pngs/train.json'), 'r') as jsonfile:
    train = json.load(jsonfile)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/find_one_find_pngs/valid_seen.json'), 'r') as jsonfile:
    valid_seen = json.load(jsonfile)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/find_one_find_pngs/valid_unseen.json'), 'r') as jsonfile:
    valid_unseen = json.load(jsonfile)

global current_trajectory_index
current_trajectory_index = 0
pbar = tqdm(total=len(valid_unseen))

lock = threading.Lock() # To protect current_trajectory_index

def run(thread_id):
    thor_env = ThorEnv()
    fo = FindOne(thor_env, obj_type_to_index)

    # trajectory index, target type, receptacle type, openable, open
    targets_receptacles = []
    skipped_trajectory_indexes = []

    global current_trajectory_index
    while True:
        lock.acquire()
        if current_trajectory_index < len(valid_unseen):
            trajectory_index = current_trajectory_index
            current_trajectory_index += 1
            '''
            print('Remaining trajectories: ' + str(len(valid_unseen) -
                current_trajectory_index))
            '''
            pbar.update(1)
            if current_trajectory_index == len(valid_unseen):
                pbar.close()
            lock.release()
        else:
            lock.release()
            break

        trajectory = valid_unseen[trajectory_index]

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
    with open(os.path.join(os.environ['ALFRED_ROOT'],
            'data/valid_unseen_hidden_target_stats_' + str(thread_id) + '.json'),
            'w+') as outfile:
        json.dump(targets_receptacles, outfile, indent=4)
    with open(os.path.join(os.environ['ALFRED_ROOT'],
            'data/valid_unseen_hidden_target_skipped_' + str(thread_id) + '.json'),
            'w+') as outfile:
        json.dump(skipped_trajectory_indexes, outfile, indent=4)
    print('Skipped trajectory indexes:')
    print(skipped_trajectory_indexes)

    thor_env.stop()

#if __name__ == '__main__':
args = parser.parse_args()


'''
threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run, args=(n,))
    threads.append(thread)
    thread.start()
    time.sleep(1)
'''

'''
total_hidden_targets = 0
hidden_target_counts = defaultdict(int)
containers_counts = defaultdict(int)
combined_hidden_stats = []
for i in range(8):
    stats_path = (os.path.join(os.environ['ALFRED_ROOT'],
        'data/valid_seen_hidden_target_stats_' + str(i) + '.json'))
    with open(stats_path, 'r') as jsonfile:
        # List of trajectory_index, target_type, parent_receptacle_type,
        # parent_receptacle is open
        stats = json.load(jsonfile)
        # Don't count toilets
        current_hidden_targets = [traj for traj in stats if traj[2] != 'Toilet']
        total_hidden_targets += len(current_hidden_targets)
        for traj_index, target, receptacle, _ in stats:
            hidden_target_counts[target] += 1
            containers_counts[receptacle] += 1
        combined_hidden_stats.extend(stats)

print(len(combined_hidden_stats))
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/find_one_find_pngs_valid_seen_hidden_object_stats.json'),
        'w') as outfile:
    json.dump(combined_hidden_stats, outfile, indent=4)

# 14K out of 100K training trajectories have a hidden target, not counting
# toilets
#
# 541 out of valid_seen
#
# 707 out of valid_unseen
#
# train
# [('Cup', 1509), ('Potato', 1495), ('Apple', 1247), ('Tomato', 1211), ('Mug',
# 1038), ('Plate', 936), ('SoapBar', 894), ('Bowl', 719), ('SprayBottle', 571),
# ('Candle', 569), ('SoapBottle', 542), ('ToiletPaper', 534), ('Cloth', 424),
# ('DishSponge', 376), ('Egg', 339), ('Knife', 322), ('Lettuce', 306),
# ('Ladle', 262), ('CD', 235), ('SaltShaker', 215), ('KeyChain', 201),
# ('CreditCard', 195), ('Bread', 187), ('Pan', 181), ('PepperShaker', 172),
# ('Spoon', 167), ('CellPhone', 153), ('ButterKnife', 149), ('Fork', 145),
# ('Pencil', 124), ('Watch', 120), ('Pen', 117), ('Kettle', 111), ('Book',
# 111), ('TissueBox', 105), ('Glassbottle', 91), ('Pot', 89), ('Spatula', 85),
# ('WineBottle', 69), ('Vase', 39), ('HandTowel', 31), ('RemoteControl', 27),
# ('Newspaper', 25), ('Box', 11)]
#
# [('Fridge', 5587), ('Cabinet', 4465), ('Drawer', 3007), ('Toilet', 2064),
# ('Microwave', 1266), ('Safe', 60)]
#
# valid_seen
# [('Apple', 85), ('Tomato', 77), ('Potato', 51), ('SoapBar', 42),
# ('SprayBottle', 38), ('Plate', 36), ('ToiletPaper', 36), ('Cloth', 27),
# ('Cup', 25), ('Lettuce', 21), ('ButterKnife', 18), ('Mug', 18), ('TissueBox',
# 15), ('Knife', 15), ('CD', 15), ('CreditCard', 15), ('Bowl', 14),
# ('DishSponge', 12), ('SoapBottle', 12), ('Egg', 9), ('KeyChain', 9),
# ('Ladle', 8), ('Pot', 6), ('CellPhone', 6), ('RemoteControl', 5), ('Pan', 3),
# ('Newspaper', 3), ('Fork', 3), ('Spatula', 3), ('Candle', 3), ('Pencil', 3)]
#
# [('Fridge', 235), ('Drawer', 141), ('Cabinet', 101), ('Toilet', 92),
# ('Microwave', 61), ('Safe', 3)]
#
# valid_unseen
# [('Mug', 93), ('Apple', 78), ('Cup', 71), ('SoapBar', 61), ('Plate', 45),
# ('PepperShaker', 39), ('Lettuce', 36), ('Cloth', 36), ('CD', 30), ('Potato',
# 30), ('KeyChain', 27), ('Vase', 21), ('ToiletPaper', 18), ('Egg', 18),
# ('Bowl', 18), ('SaltShaker', 15), ('TissueBox', 15), ('Bread', 12),
# ('Newspaper', 12), ('Tomato', 12), ('Glassbottle', 9), ('Pan', 9), ('Watch',
# 9), ('Pencil', 9), ('CreditCard', 9), ('SoapBottle', 6), ('Box', 6),
# ('ButterKnife', 3), ('CellPhone', 3), ('Knife', 3), ('Fork', 3)]
#
# [('Cabinet', 327), ('Fridge', 246), ('Drawer', 81), ('Toilet', 49), ('Safe',
# 36), ('Microwave', 17)] 0%|
print('total hidden targets: ' + str(total_hidden_targets))
sorted_hidden_target_counts = sorted(hidden_target_counts.items(), key=lambda x:x[1], reverse=True)
print('hidden target counts: ')
print(sorted_hidden_target_counts)

sorted_containers_counts = sorted(containers_counts.items(), key=lambda x:x[1], reverse=True)
print('containers counts: ')
print(sorted_containers_counts)
'''

'''
counts = defaultdict(int)

with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/find_one_find_pngs/train.json'), 'r') as jsonfile:
    train = json.load(jsonfile)

for trajectory in tqdm(train):
    counts[trajectory['target']] += 1

sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)

print('total trajectories: ' + str(len(train)))

# 101295 training trajectories
print('target counts: ')
print(sorted_counts)
'''

'''
thor_env = ThorEnv()
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

with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/scenes_contained_objects.json'), 'w') as outfile:
    json.dump(contained_objects, outfile, indent=4)

# Lightly analyze the objects and containers
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'data/scenes_contained_objects.json'), 'r') as jsonfile:
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
thor_env = ThorEnv()
cookable = defaultdict(int)
canChangeTempToHot = defaultdict(int)
cookable_not_canChangeTempToHot = defaultdict(int)
canChangeTempToHot_not_cookable = defaultdict(int)
for scene_num in tqdm(constants.SCENE_NUMBERS):
    thor_env.reset(scene_num)
    for obj in thor_env.last_event.metadata['objects']:
        if obj['cookable']:
            cookable[obj['objectType']] += 1
        elif obj['canChangeTempToHot']:
            canChangeTempToHot[obj['objectType']] += 1

        if obj['cookable'] and not obj['canChangeTempToHot']:
            cookable_not_canChangeTempToHot[obj['objectType']] += 1
        elif obj['canChangeTempToHot'] and not obj['cookable']:
            canChangeTempToHot_not_cookable[obj['objectType']] += 1
print('cookable:')
print(cookable)
print('canChangeTempToHot:')
print(canChangeTempToHot)

print('cookable not canChangeTempToHot:')
print(cookable_not_canChangeTempToHot)
print('canChangeTempToHot not cookable:')
print(canChangeTempToHot_not_cookable)
