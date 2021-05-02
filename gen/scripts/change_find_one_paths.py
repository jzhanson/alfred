import os
import json

for dataset in ['find_one', 'find_one_find', 'find_one_find_pngs', 'find_one_pngs']:
    for split in ['train', 'valid_seen', 'valid_unseen']:
        json_path = os.path.join('/home/jzhanson/alfred/data', dataset, split + '.json')
        with open(json_path, 'r') as jsonfile:
            trajectories = json.load(jsonfile)

        for trajectory in trajectories:
            path_suffix = trajectory['path'].split('/home/jzhanson/')[1]
            new_path = os.path.join('/data/datasets/', path_suffix)
            if not os.path.isdir(new_path):
                print(new_path + ' does not exist!')
            trajectory['path'] = new_path

        outpath = os.path.join('/home/jzhanson/alfred/data', dataset, split + '_data.json')
        with open(outpath, 'w') as outfile:
            json.dump(trajectories, outfile, indent=4)

for dataset in ['find_one', 'find_one_find', 'find_one_find_pngs', 'find_one_pngs']:
    for split in ['train', 'valid_seen', 'valid_unseen']:
        json_path = os.path.join('/home/jzhanson/alfred/data', dataset, split + '.json')
        new_json_path = os.path.join('/home/jzhanson/alfred/data', dataset, split + '_data.json')
        os.system('mv ' + new_json_path + ' ' + json_path)
