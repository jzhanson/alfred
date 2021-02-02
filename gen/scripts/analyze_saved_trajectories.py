import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
'''
import matplotlib as mpl
mpl.use('Agg')
'''
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import constants

parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model-path', type=str, default='', nargs='+', help='path to model directory')
parser.add_argument('-o', '--outpath', type=str, default='../..', help='path to output images')

splits = ['train', 'valid_seen', 'valid_unseen']
trajectory_online = ['trajectory', 'online']

# Stuff that can be put inside a receptacle
SMALL_OBJECTS = []
for receptacle, objects in constants.VAL_RECEPTACLE_OBJECTS.items():
    SMALL_OBJECTS.extend(list(objects))
SMALL_OBJECTS = set(SMALL_OBJECTS)
# Stuff that can't be put inside a receptacle
LARGE_OBJECTS = set(constants.OBJECTS) - SMALL_OBJECTS
'''
with open('/home/jzhanson/alfred/data/find_one_find_pngs/obj_type_to_index.json') as jsonfile:
    obj_type_to_index = json.load(jsonfile)
print(set(obj_type_to_index.keys()) - SMALL_OBJECTS - LARGE_OBJECTS)
'''
small_large = ['small', 'large']

print(SMALL_OBJECTS)
print(LARGE_OBJECTS)

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)

    if type(args.model_path) is str:
        model_paths = [args.model_path]
    elif type(args.model_path) is list:
        model_paths = args.model_path

    for model_path in model_paths:
        print(model_path)

        checkpoint_dirs = [f for f in os.listdir(model_path) if
                os.path.isdir(os.path.join(model_path, f)) and f !=
                'tensorboard_logs']

        # checkpoint, trajectory or online, train or valid_seen or valid_unseen,
        # target, crow_distance, target_visible, initial_crow_distance, small or
        # large object
        pre_df = []

        for checkpoint_dir in checkpoint_dirs:
            trajectory_checkpoint_path = os.path.join(model_path,
                    str(checkpoint_dir))
            for trajectory_or_online in trajectory_online:
                try:
                    # In case some trajectory evals were not completed
                    with open(os.path.join(trajectory_checkpoint_path,
                        trajectory_or_online + '.json')) as jsonfile:
                        saved_trajectories = json.load(jsonfile)
                except:
                    continue

                for split in splits:
                    for i in range(20):
                        pre_df.append((int(checkpoint_dir), trajectory_or_online,
                            split, saved_trajectories[split]['target'][i],
                            saved_trajectories[split]['crow_distance_to_goal'][i],
                            saved_trajectories[split]['initial_crow_distance'][i],
                            saved_trajectories[split]['target_visible'][i], 'small'
                            if saved_trajectories[split]['target'][i] in
                            SMALL_OBJECTS else 'large'
                        ))
        df = pd.DataFrame(data=pre_df, columns=['checkpoint',
            'trajectory_or_online', 'split', 'target', 'crow_distance',
            'initial_crow_distance', 'target_visible', 'target_size'])

        for trajectory_or_online in trajectory_online:
            print(trajectory_or_online + ' small targets: %d large targets: %d'
                    % (len(df[np.logical_and(df['target_size'] == 'small',
                        df['trajectory_or_online'] == trajectory_or_online)]),
                        len(df[np.logical_and(df['target_size'] == 'large',
                            df['trajectory_or_online'] ==
                            trajectory_or_online)])))

        sns.relplot(kind='line', x='checkpoint', y='crow_distance',
                hue='target_size', row='split', col='trajectory_or_online',
                row_order=splits, col_order=trajectory_online,
                hue_order=small_large, data=df, facet_kws={'margin_titles' : True})

        plt.subplots_adjust(top=0.9)
        model_name = model_path.split('/')[-1]
        plt.gcf().suptitle(model_name)
        plt.savefig(os.path.join(args.outpath, model_name + '_distances.jpg'))
        plt.close()

