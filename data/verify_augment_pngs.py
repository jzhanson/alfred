import os
import sys
import glob
import shutil
from tqdm import tqdm

skipped_files = ['data/full_2.1.0/train/pick_and_place_with_movable_recep-Pen-Bowl-Dresser-311/trial_T20190908_170820_174380/traj_data.json',

        'data/full_2.1.0/train/pick_cool_then_place_in_recep-AppleSliced-None-DiningTable-27/trial_T20190907_171803_405680/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141/traj_data.json',
        'data/full_2.1.0/train/look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182531_510491/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-26/trial_T20190908_162237_908840/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-23/trial_T20190907_123248_978930/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T20190907_151643_465634/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-CounterTop-19/trial_T20190909_053101_102010/traj_data.json',
        'data/full_2.1.0/train/pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-LettuceSliced-Pot-DiningTable-21/trial_T20190907_160923_689765/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-DiningTable-24/trial_T20190908_194409_961394/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-AppleSliced-None-SideTable-3/trial_T20190908_110347_206140/traj_data.json',
        'data/full_2.1.0/train/look_at_obj_in_light-Pen-None-DeskLamp-316/trial_T20190908_061814_700195/traj_data.json',
        'data/full_2.1.0/train/pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-AppleSliced-None-SinkBasin-4/trial_T20190907_154556_101106/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539/traj_data.json',
        'data/full_2.1.0/train/look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182720_056041/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Lettuce-None-SinkBasin-23/trial_T20190908_173530_026785/traj_data.json']

new_skipped_files = ['data/full_2.1.0/train/pick_cool_then_place_in_recep-Lettuce-None-SinkBasin-23/trial_T20190908_173530_026785/traj_data.json',
        'data/full_2.1.0/train/look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182720_056041/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167/traj_data.json',
        'data/full_2.1.0/train/pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388/traj_data.json',
        'data/full_2.1.0/train/look_at_obj_in_light-Pen-None-DeskLamp-316/trial_T20190908_061814_700195/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-AppleSliced-None-SideTable-3/trial_T20190908_110347_206140/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-DiningTable-24/trial_T20190908_194409_961394/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-LettuceSliced-Pot-DiningTable-21/trial_T20190907_160923_689765/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630/traj_data.json',
        'data/full_2.1.0/train/pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-CounterTop-19/trial_T20190909_053101_102010/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T20190907_151643_465634/traj_data.json',
        'data/full_2.1.0/train/pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-23/trial_T20190907_123248_978930/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-26/trial_T20190908_162237_908840/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614/traj_data.json',
        'data/full_2.1.0/train/look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182531_510491/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433/traj_data.json',
        'data/full_2.1.0/train/pick_cool_then_place_in_recep-AppleSliced-None-DiningTable-27/trial_T20190907_171803_405680/traj_data.json',
        'data/full_2.1.0/train/pick_and_place_with_movable_recep-Pen-Bowl-Dresser-311/trial_T20190908_170820_174380/traj_data.json']

'''
def get_image_index(save_path):
    num_images = len(glob.glob(save_path + '/*.png'))
    # Downloaded ALFRED dataset (from Mohit) contains jpgs
    if num_images == 0:
        num_images = len(glob.glob(save_path + '/*.jpg'))
    return num_images

# Check whether any skipped files still has high_res_images directory
for skipped_path in new_skipped_files:
    directory = '/'.join(skipped_path.split('/')[:-1])
    high_res_path = os.path.join(directory, 'high_res_images')
    raw_path = os.path.join(directory, 'raw_images')
    if os.path.isdir(high_res_path):
        print(directory + ' high_res_images: ' + str(get_image_index(high_res_path)) + ' raw_images: ' + str(get_image_index(raw_path)))
        shutil.rmtree(high_res_path)



print('number of skipped files: ' + str(len(skipped_files)))
print('number of skipped files: ' + str(len(new_skipped_files)))
'''

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

raw_images_bytes = 0
removable_raw_images_bytes = 0
high_res_images_bytes = 0
#features_bytes = 0
for split in ['train', 'valid_seen', 'valid_unseen']:
    split_path = os.path.join('data/full_2.1.0', split)
    for task_spec in tqdm(os.listdir(split_path)):
            task_spec_path = os.path.join(split_path, task_spec)
            for trial in os.listdir(task_spec_path):
                trial_path = os.path.join(task_spec_path, trial)
                high_res_images_path = os.path.join(trial_path, 'high_res_images')

                # Move high_res_images to shared dataset
                destination_path = os.path.join(
                        '/data/datasets/alfred_high_res_images/data/full_2.1.0', split,
                        task_spec, trial)
                if not os.path.isdir(destination_path):
                    print('destination path not found! ' + destination_path)
                    continue
                elif os.path.isdir(os.path.join(destination_path, 'high_res_images')):
                    print('destination path already exists! ' +
                            os.path.join(destination_path, 'high_res_images'))
                else:
                    os.system('mv -i ' + high_res_images_path + ' ' + destination_path)

                '''
                # Count bytes
                raw_images_path = os.path.join(trial_path, 'raw_images')
                raw_images_bytes += sum(os.path.getsize(os.path.join(raw_images_path, f)) for f in os.listdir(raw_images_path))
                #features_bytes += os.path.getsize(os.path.join(trial_path, 'feat_conv.pt'))
                if os.path.isdir(high_res_images_path):
                    removable_raw_images_bytes += sum(os.path.getsize(os.path.join(raw_images_path, f)) for f in os.listdir(raw_images_path))
                    high_res_images_bytes += sum(os.path.getsize(os.path.join(high_res_images_path, f)) for f in os.listdir(high_res_images_path))
                # Delete dataset resnet features
                #os.remove(os.path.join(trial_path, 'feat_conv.pt'))

print('raw_images_bytes: ' + sizeof_fmt(raw_images_bytes))
print('removable_raw_images_bytes: ' + sizeof_fmt(removable_raw_images_bytes))
print('high_res_images_bytes: ' + sizeof_fmt(high_res_images_bytes))
#print('features_bytes: ' + sizeof_fmt(features_bytes))

                '''
