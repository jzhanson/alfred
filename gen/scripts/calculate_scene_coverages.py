import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
import gen.constants as constants
from gen.graph.graph_obj import Graph

import json

from tqdm import tqdm

import cv2
from env.thor_env import ThorEnv

scene_numbers = (list(constants.SCENE_TYPE['Kitchen']) +
        list(constants.SCENE_TYPE['LivingRoom']) +
        list(constants.SCENE_TYPE['Bedroom']) +
        list(constants.SCENE_TYPE['Bathroom']))

thor_env = ThorEnv()

scene_navigation_coverages = {}
scene_interaction_coverages = {}
scene_state_change_coverages = {}
for scene_number in tqdm(scene_numbers):
    event = thor_env.reset(scene_number)
    graph = Graph(use_gt=True, construct_graph=True, scene_id=scene_number)

    scene_navigation_coverages[scene_number] = len(graph.points)

    has_faucet_sinkbasin = any([obj['objectType'] == 'SinkBasin' for obj in
        event.metadata['objects']]) and any([obj['objectType'] == 'Faucet' for
            obj in event.metadata['objects']])
    has_knife = any(['Knife' in obj['objectType'] for obj in
        event.metadata['objects']])
    has_microwave = any([obj['objectType'] == 'Microwave' for obj in
        event.metadata['objects']])
    has_fridge = any([obj['objectType'] == 'Fridge' for obj in
        event.metadata['objects']])

    interaction_coverage = 0
    state_change_coverage = 0
    for obj in event.metadata['objects']:
        if obj['toggleable']:
            interaction_coverage += 2 # ToggleObjectOn, ToggleObjectOff
        if obj['openable']:
            interaction_coverage += 2 # OpenObject, CloseObject
        if obj['pickupable']:
            interaction_coverage += 2 # PickupObject, PutObject
        if obj['sliceable'] and has_knife:
            interaction_coverage += 1 # SliceObject

        # These state_change coverages are based on env/thor_env.py
        # Only count cleaned state change affordance if there is a Faucet and
        # SinkBasin in the scene
        if obj['dirtyable'] and obj['isDirty'] and has_faucet_sinkbasin:
            state_change_coverage += 1 # cleaned_objects
        # Anything that fits in the microwave or refrigerator can be heated or
        # cooled, respectively. This only counts the object once, even if it
        # can be sliced (i.e. Apple and that same instance but as AppleSliced
        # count as one state change)
        if (obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS['Microwave']
                and has_microwave):
            state_change_coverage += 1
        if (obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS['Fridge'] and
                has_fridge):
            state_change_coverage += 1
    scene_interaction_coverages[scene_number] = interaction_coverage
    scene_state_change_coverages[scene_number] = state_change_coverage

print(scene_navigation_coverages)
print(scene_interaction_coverages)
print(scene_state_change_coverages)

with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_navigation_coverages.json'), 'w') as jsonfile:
    json.dump(scene_navigation_coverages, jsonfile, indent=4)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_interaction_coverages.json'), 'w') as jsonfile:
    json.dump(scene_interaction_coverages, jsonfile, indent=4)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_state_change_coverages.json'), 'w') as jsonfile:
    json.dump(scene_state_change_coverages, jsonfile, indent=4)

'''
# Basic statistics and graphs
import numpy as np
navigation_coverages = np.zeros(len(scene_numbers))
interaction_coverages = np.zeros(len(scene_numbers))
state_change_coverages = np.zeros(len(scene_numbers))
for scene_number in range(1, 31):
    navigation_coverages[scene_number - 1] = constants.SCENE_NAVIGATION_COVERAGES[scene_number]
    interaction_coverages[scene_number - 1] = constants.SCENE_INTERACTION_COVERAGES[scene_number]
    state_change_coverages[scene_number - 1] = constants.SCENE_STATE_CHANGE_COVERAGES[scene_number]
for scene_number in range(201, 231):
    navigation_coverages[scene_number - 201 + 30] = constants.SCENE_NAVIGATION_COVERAGES[scene_number]
    interaction_coverages[scene_number - 201 + 30] = constants.SCENE_INTERACTION_COVERAGES[scene_number]
    state_change_coverages[scene_number - 201 + 30] = constants.SCENE_STATE_CHANGE_COVERAGES[scene_number]
for scene_number in range(301, 331):
    navigation_coverages[scene_number - 301 + 60] = constants.SCENE_NAVIGATION_COVERAGES[scene_number]
    interaction_coverages[scene_number - 301 + 60] = constants.SCENE_INTERACTION_COVERAGES[scene_number]
    state_change_coverages[scene_number - 301 + 60] = constants.SCENE_STATE_CHANGE_COVERAGES[scene_number]
for scene_number in range(401, 431):
    navigation_coverages[scene_number - 401 + 90] = constants.SCENE_NAVIGATION_COVERAGES[scene_number]
    interaction_coverages[scene_number - 401 + 90] = constants.SCENE_INTERACTION_COVERAGES[scene_number]
    state_change_coverages[scene_number - 401 + 90] = constants.SCENE_STATE_CHANGE_COVERAGES[scene_number]

print('kitchen navigation mean/std', np.mean(navigation_coverages[:30]), np.std(navigation_coverages[:30]))
print('living room navigation mean/std', np.mean(navigation_coverages[30:60]), np.std(navigation_coverages[30:60]))
print('bedroom navigation mean/std', np.mean(navigation_coverages[60:90]), np.std(navigation_coverages[60:90]))
print('bathroom navigation mean/std', np.mean(navigation_coverages[90:120]), np.std(navigation_coverages[90:120]))

print('kitchen interaction mean/std', np.mean(interaction_coverages[:30]), np.std(interaction_coverages[:30]))
print('living room interaction mean/std', np.mean(interaction_coverages[30:60]), np.std(interaction_coverages[30:60]))
print('bedroom interaction mean/std', np.mean(interaction_coverages[60:90]), np.std(interaction_coverages[60:90]))
print('bathroom interaction mean/std', np.mean(interaction_coverages[90:120]), np.std(interaction_coverages[90:120]))

print('kitchen state_change mean/std', np.mean(state_change_coverages[:30]), np.std(state_change_coverages[:30]))
print('living room state_change mean/std', np.mean(state_change_coverages[30:60]), np.std(state_change_coverages[30:60]))
print('bedroom state_change mean/std', np.mean(state_change_coverages[60:90]), np.std(state_change_coverages[60:90]))
print('bathroom state_change mean/std', np.mean(state_change_coverages[90:120]), np.std(state_change_coverages[90:120]))

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.clf()
plt.boxplot([navigation_coverages[:30], navigation_coverages[30:60], navigation_coverages[60:90], navigation_coverages[90:], navigation_coverages], labels=['Kitchen', 'Living Room', 'Bedroom', 'Bathroom', 'All'])
plt.title('Scene navigation coverages')
plt.savefig(os.path.join(os.environ['ALFRED_ROOT'], 'saved/scene_navigation_coverages.png'))

plt.clf()
plt.boxplot([interaction_coverages[:30], interaction_coverages[30:60], interaction_coverages[60:90], interaction_coverages[90:], interaction_coverages], labels=['Kitchen', 'Living Room', 'Bedroom', 'Bathroom', 'All'])
plt.title('Scene interaction coverages')
plt.savefig(os.path.join(os.environ['ALFRED_ROOT'], 'saved/scene_interaction_coverages.png'))

plt.clf()
plt.boxplot([state_change_coverages[:30], state_change_coverages[30:60], state_change_coverages[60:90], state_change_coverages[90:], state_change_coverages], labels=['Kitchen', 'Living Room', 'Bedroom', 'Bathroom', 'All'])
plt.title('Scene state change coverages')
plt.savefig(os.path.join(os.environ['ALFRED_ROOT'], 'saved/scene_state_change_coverages.png'))
'''
