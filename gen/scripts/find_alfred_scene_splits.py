import json

splits = json.load(open('data/splits/oct21.json', 'r'))

for split in splits.keys():
    scene_numbers = set()
    for example in splits[split]:
        if 'test' in split:
            scene_number = 0
        else:
            scene_number = int(example['task'].split('/')[0].split('-')[-1])
        scene_numbers.add(scene_number)
    print(split)
    print(sorted(scene_numbers))

