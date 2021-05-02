import json
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

stats = []
pre_dataframe = []
for dataset in ['find_one', 'find_one_find']:
    for split in ['train', 'valid_seen', 'valid_unseen']:
        with open('data/' + dataset + '/' + split + '.json', 'r') as jsonfile:
            trajectories = json.load(jsonfile)
            stats.append((dataset, split, len(trajectories), np.mean([len(traj['low_actions']) for traj in trajectories]), np.std([len(traj['low_actions']) for traj in trajectories])))
            '''
            for num_actions in [len(traj['low_actions']) for traj in trajectories]:
                pre_dataframe.append(('gotolocation' if dataset == 'find_one' else 'augmented', split, num_actions))
            '''
            targets_dict = {}
            for target in [traj['target'] for traj in trajectories]:
                if target not in targets_dict:
                    targets_dict[target] = 1
                else:
                    targets_dict[target] += 1
            for target, occurrences in targets_dict.items():
                if occurrences > 2500: print(target)
                pre_dataframe.append(('gotolocation' if dataset == 'find_one'
                    else 'augmented', split, target, occurrences))

df = pd.DataFrame(pre_dataframe, columns=['dataset', 'split', 'target', 'target_occurrences'])

for split in ['train', 'valid_seen', 'valid_unseen']:
    plt.clf()
    g = sns.displot(
            df[df['split'] == split], x="target_occurrences", col="dataset",
            height=3, facet_kws=dict(margin_titles=True, sharex=False, sharey=False),
            )
    plt.savefig('./find_one_find_target_stats_' + split + '.png')
'''
g.axes[0,0].set_xlim(0,10)
g.axes[0,1].set_xlim(0,10)
g.axes[1,0].set_xlim(0,10)
g.axes[1,1].set_xlim(0,10)
g.axes[2,0].set_xlim(0,10)
g.axes[2,1].set_xlim(0,10)
'''



'''
df = pd.DataFrame(pre_dataframe, columns=['dataset', 'split', 'num_actions'])

plt.clf()
g = sns.displot(
        df, x="num_actions", col="dataset", row="split",
        binwidth=3, height=3, facet_kws=dict(margin_titles=True, sharey=False),
        )
g.axes[0,0].set_ylim(0,20000)
g.axes[0,1].set_ylim(0,20000)
g.axes[1,0].set_ylim(0,1000)
g.axes[1,1].set_ylim(0,1000)
g.axes[2,0].set_ylim(0,1000)
g.axes[2,1].set_ylim(0,1000)

plt.savefig('./find_one_find_dataset_stats.png')

print(stats)
'''

