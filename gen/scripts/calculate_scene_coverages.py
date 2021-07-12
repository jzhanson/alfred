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
scene_interaction_coverages_by_object = {}
scene_state_change_coverages_by_object = {}
scene_interaction_coverages_by_type = {}
scene_state_change_coverages_by_type = {}
all_object_types = set()
kitchen_object_types = set()
living_room_object_types = set()
bedroom_object_types = set()
bathroom_object_types = set()

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

    seen_object_types_in_scene = set()

    # Another note: coverage can be weird for the single_interact case.
    # I'm choosing not to implement coverage for single_interact because the
    # InteractionExploration environment chooses the complex action and
    # InteractionReward is only aware of the actual complex action taken. Also,
    # single interact is meant to be a learning/sanity check crutch anyways,
    # not a final evaluation, due to the finicky-ness of choosing a contextual
    # interact action.
    interaction_coverage_by_object = 0
    interaction_coverage_by_type = 0
    state_change_coverage_by_object = 0
    state_change_coverage_by_type = 0
    for obj in event.metadata['objects']:
        all_object_types.add(obj['objectType'])
        if scene_number in list(constants.SCENE_TYPE['Kitchen']):
            kitchen_object_types.add(obj['objectType'])
        elif scene_number in list(constants.SCENE_TYPE['LivingRoom']):
            living_room_object_types.add(obj['objectType'])
        elif scene_number in list(constants.SCENE_TYPE['Bedroom']):
            bedroom_object_types.add(obj['objectType'])
        elif scene_number in list(constants.SCENE_TYPE['Bathroom']):
            bathroom_object_types.add(obj['objectType'])

        object_type_seen_before = (obj['objectType'] in
                seen_object_types_in_scene)
        # Get interaction coverages by object (e.g. two instances of Fork both
        # count their valid interactions towards coverage)
        if obj['toggleable']:
            # ToggleObjectOn, ToggleObjectOff
            interaction_coverage_by_object += 2
            if not object_type_seen_before:
                interaction_coverage_by_type += 2
        if obj['openable']:
            interaction_coverage_by_object += 2 # OpenObject, CloseObject
            if not object_type_seen_before:
                interaction_coverage_by_type += 2
        if obj['pickupable']:
            interaction_coverage_by_object += 2 # PickupObject, PutObject
            if not object_type_seen_before:
                interaction_coverage_by_type += 2
        if obj['sliceable'] and has_knife:
            interaction_coverage_by_object += 1 # SliceObject
            single_slice_interaction_coverage_by_object = 0
            if not object_type_seen_before:
                interaction_coverage_by_type += 1
            obj_slice_type = obj['objectType'] + 'Sliced'
            # All slices are pickupable, and not toggleable, openable, or
            # sliceable
            single_slice_interaction_coverage_by_object += 2
            single_slice_state_change_coverage_by_object = 0
            if (obj_slice_type in constants.VAL_RECEPTACLE_OBJECTS['SinkBasin']
                    and has_faucet_sinkbasin):
                single_slice_state_change_coverage_by_object += 1
            if (obj_slice_type in constants.VAL_RECEPTACLE_OBJECTS['Microwave']
                    and has_microwave):
                single_slice_state_change_coverage_by_object += 1
            if (obj_slice_type in constants.VAL_RECEPTACLE_OBJECTS['Fridge']
                    and has_fridge):
                single_slice_state_change_coverage_by_object += 1
            # Assuming all slices have the same properties even though one
            # slice is still the same object type for Bread, Lettuce, and maybe
            # others. AFAIK this is the case
            interaction_coverage_by_object += (
                    single_slice_interaction_coverage_by_object *
                    constants.NUM_SLICED_OBJ_PARTS[obj['objectType']])
            state_change_coverage_by_object += (
                    single_slice_state_change_coverage_by_object *
                    constants.NUM_SLICED_OBJ_PARTS[obj['objectType']])
            if not obj_slice_type in seen_object_types_in_scene:
                interaction_coverage_by_type += (
                        single_slice_interaction_coverage_by_object)
                state_change_coverage_by_type += (
                        single_slice_state_change_coverage_by_object)
                seen_object_types_in_scene.add(obj_slice_type)

        # TODO: It is possible that some objects in the scene would be able to
        # fit into these receptacles and be cleaned/heated/cooled but don't
        # show up in constants.VAL_RECEPTACLE_OBJECTS - is this a problem?

        # Only count cleaned state change affordance if there is a Faucet and
        # SinkBasin in the scene and object fits in the sink
        if (obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS['SinkBasin']
                and has_faucet_sinkbasin):
            state_change_coverage_by_object += 1 # cleaned_objects
            if not object_type_seen_before:
                state_change_coverage_by_type += 1
        # Anything that fits in the microwave or refrigerator can be heated or
        # cooled, respectively. This only counts the object once, even if it
        # can be sliced (i.e. Apple and that same instance but as AppleSliced
        # count as one state change)
        if (obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS['Microwave']
                and has_microwave):
            state_change_coverage_by_object += 1
            if not object_type_seen_before:
                state_change_coverage_by_type += 1
        if (obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS['Fridge'] and
                has_fridge):
            state_change_coverage_by_object += 1
            if not object_type_seen_before:
                state_change_coverage_by_type += 1

        if not object_type_seen_before:
            seen_object_types_in_scene.add(obj['objectType'])

    scene_interaction_coverages_by_object[scene_number] = (
            interaction_coverage_by_object)
    scene_state_change_coverages_by_object[scene_number] = (
            state_change_coverage_by_object)
    scene_interaction_coverages_by_type[scene_number] = (
            interaction_coverage_by_type)
    scene_state_change_coverages_by_type[scene_number] = (
            state_change_coverage_by_type)

#print(all_object_types)
print('kitchen_object_types', kitchen_object_types)
print('living_room_object_types', living_room_object_types)
print('bedroom_object_types', bedroom_object_types)
print('bathroom_object_types', bathroom_object_types)

print('unique kitchen types', kitchen_object_types - living_room_object_types.union(bedroom_object_types).union(bathroom_object_types))
print('unique living room types', living_room_object_types - kitchen_object_types.union(bedroom_object_types).union(bathroom_object_types))
print('unique bedroom types', bedroom_object_types - kitchen_object_types.union(living_room_object_types).union(bathroom_object_types))
print('unique bathroom types', bathroom_object_types - kitchen_object_types.union(living_room_object_types).union(bedroom_object_types))

'''
print(scene_navigation_coverages)
print(scene_interaction_coverages_by_object)
print(scene_state_change_coverages_by_object)
print(scene_interaction_coverages_by_type)
print(scene_state_change_coverages_by_type)

with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_navigation_coverages.json'), 'w') as jsonfile:
    json.dump(scene_navigation_coverages, jsonfile, indent=4)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_interaction_coverages_by_object.json'), 'w') as jsonfile:
    json.dump(scene_interaction_coverages_by_object, jsonfile, indent=4)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_state_change_coverages_by_object.json'), 'w') as jsonfile:
    json.dump(scene_state_change_coverages_by_object, jsonfile, indent=4)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_interaction_coverages_by_type.json'), 'w') as jsonfile:
    json.dump(scene_interaction_coverages_by_type, jsonfile, indent=4)
with open(os.path.join(os.environ['ALFRED_ROOT'],
        'scene_state_change_coverages_by_type.json'), 'w') as jsonfile:
    json.dump(scene_state_change_coverages_by_type, jsonfile, indent=4)
'''

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
