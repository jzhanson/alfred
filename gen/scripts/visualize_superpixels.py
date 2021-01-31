import os
import sys
import shutil
# From https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outpath', type=str, default='../../superpixel_visualizations', help='path to output directory')
parser.add_argument('-stp', '--selected-trajectories-path', type=str,
        default='../../selected_trajectories',
        help='path to selected trajectories, with same structure as ALFRED dataset but without raw_images directory (trajectory directory, then trial directory, then images)')
parser.add_argument('-dr', '--dry-run', dest='dry_run', action='store_true',
        help='whether to only save a single image called sample_superpixel.png')
parser.set_defaults(dry_run=False)

# Approx number of superpixels to create
# 10, 25, 50, 100
parser.add_argument('-ns', '--num-segments', type=int, nargs='+', default=10,
        help='numbers of segments')
# Balances color proximity and space proximity. Higher values give more weight
# to space proximity, making superpixel shapes more square/cubic
# Can try on a log scale: 0.1, 1.0, 10.0, or constants: 1.0, 5.0, 10.0, 15.0
parser.add_argument('-c', '--compactnesses', type=float, nargs='+',
        default=10.0, help='compactnesses to try')
# Pre-processing Gaussian kernel size for smoothing --- can be tuples for each
# dimension
# 0 means no smoothing
# 0, 1, 2, 5, 10
parser.add_argument('-s', '--sigmas', type=int, nargs='+', default=0,
        help='sigmas to try')
# "Proportion of the minimum segment size to be removed with respect to the
# supposed segment size `depth*width*height/n_segments`"
# i.e. the smaller the value, the smaller segments are allowed
# default from scikit-image is 0.5, can try 0.1, 0.05, but I like 0.01, 0.005,
# 0.001
parser.add_argument('-msf', '--min-size-factors', type=float, nargs='+',
        default=0.01, help='min_size_factors to try')

parser.add_argument('-or', '--outer-rows', type=str,
        default='compactness', help='parameter name to iterate over outer rows')
parser.add_argument('-oc', '--outer-cols', type=str,
        default='n_segments', help='parameter name to iterate over outer columns')
parser.add_argument('-ir', '--inner-rows', type=str,
        default='min_size_factor', help='parameter name to iterate over inner rows')
parser.add_argument('-ic', '--inner-cols', type=str,
        default='sigma', help='parameter name to iterate over inner columns')

'''
#trajectory_subpaths = ['pick_two_obj_and_place-WineBottle-None-GarbageCan-1/trial_T20190906_200307_446636']
trajectory_subpaths = [
        'pick_cool_then_place_in_recep-Potato-None-CounterTop-14/trial_T20190908_053512_113312',
        'look_at_obj_in_light-BaseballBat-None-DeskLamp-303/trial_T20190907_060429_471715',
        'look_at_obj_in_light-Box-None-FloorLamp-205/trial_T20190906_211850_157561',
        'pick_cool_then_place_in_recep-Cup-None-Cabinet-12/trial_T20190909_102554_108303',
        'look_at_obj_in_light-CreditCard-None-FloorLamp-227/trial_T20190908_173638_377422',
        'pick_two_obj_and_place-TissueBox-None-SideTable-321/trial_T20190908_025939_647835',
        'pick_heat_then_place_in_recep-BreadSliced-None-SideTable-21/trial_T20190907_211124_867772',
        'look_at_obj_in_light-Pencil-None-DeskLamp-309/trial_T20190906_203043_983029',
        'pick_clean_then_place_in_recep-Bowl-None-Microwave-30/trial_T20190907_063817_005458',
        'look_at_obj_in_light-Laptop-None-FloorLamp-211/trial_T20190908_115759_129219',
        'pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-1/trial_T20190907_222837_842651',
        'pick_two_obj_and_place-ToiletPaper-None-Toilet-405/trial_T20190909_065947_451268',
        'pick_two_obj_and_place-WineBottle-None-GarbageCan-1/trial_T20190906_200307_446636',
        'pick_two_obj_and_place-Watch-None-SideTable-217/trial_T20190909_122056_142455',
        'pick_clean_then_place_in_recep-ButterKnife-None-DiningTable-16/trial_T20190909_114435_692001'
        ]
'''

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)
    else:
        shutil.rmtree(args.outpath)
        os.makedirs(args.outpath)

    # Build a list of trajectories (technically trials) to find images in
    trajectory_subpaths = []
    for trajectory_dir in os.listdir(args.selected_trajectories_path):
        trajectory_path = os.path.join(args.selected_trajectories_path,
                trajectory_dir)
        if os.path.isdir(trajectory_path):
            for trial_dir in os.listdir(trajectory_path):
                trial_path = os.path.join(trajectory_path, trial_dir)
                if os.path.isdir(trial_path):
                    trajectory_subpaths.append(os.path.join(trajectory_dir,
                        trial_dir))

    print('trajectories: ' + str(len(trajectory_subpaths)))

    # Get the paths of each image and create output directories for each
    # trajectory+trial
    image_paths = []
    for trajectory_subpath in trajectory_subpaths:
        images_path = os.path.join(args.selected_trajectories_path,
                trajectory_subpath)
        for image_name in os.listdir(images_path):
            if ('.jpg' in image_name) or ('.png' in image_name) or ('.jpeg' in
                    image_name):
                image_paths.append(os.path.join(images_path, image_name))
        if not args.dry_run:
            trajectory_outpath = os.path.join(args.outpath, trajectory_subpath)
            if not os.path.isdir(trajectory_outpath):
                os.makedirs(trajectory_outpath)

    print('images: ' + str(len(image_paths)))

    params = {
        'n_segments' :  [args.num_segments] if type(args.num_segments) is
            int else args.num_segments,
        'sigma' : [args.sigmas] if type(args.sigmas) is int else
            args.sigmas,
        'compactness' : [args.compactnesses] if type(args.compactnesses)
            is float else args.compactnesses,
        'min_size_factor' : [args.min_size_factors] if
            type(args.min_size_factors) is float else args.min_size_factors
    }


    # Iterate over image_paths instead of double loop over trajectory_subpaths
    # and image paths within that just so we can get a simple tqdm bar
    for image_path in tqdm(image_paths):
        image = img_as_float(io.imread(image_path))
        #image =  img_as_float(io.imread('/home/jzhanson/alfred/selected_trajectories/pick_two_obj_and_place-WineBottle-None-GarbageCan-1/trial_T20190906_200307_446636/000000121.jpg'))
        #image =  img_as_float(io.imread('/home/jzhanson/alfred/selected_trajectories/pick_cool_then_place_in_recep-Cup-None-Cabinet-12/trial_T20190909_102554_108303/000000181.jpg'))
        #image =  img_as_float(io.imread('/home/jzhanson/alfred/selected_trajectories/pick_cool_then_place_in_recep-Cup-None-Cabinet-12/trial_T20190909_102554_108303/000000073.jpg'))

        # Top row only displays original image
        nrows = len(params[args.outer_rows])*len(params[args.inner_rows])+1
        ncols = len(params[args.outer_cols])*len(params[args.inner_cols])

        fig, ax = plt.subplots(nrows=nrows,
                ncols=ncols, squeeze=False, figsize=(ncols*3,nrows*3),
                sharex='all', sharey='all')
        # Display original image and clear other subplots on the same row
        ax[0][0].imshow(image)
        ax[0][0].axis('off')
        for i in range(1, ncols):
            ax[0][i].remove()
        for outer_col in range(len(params[args.outer_cols])):
            for outer_row in range(len(params[args.outer_rows])):
                for row in range(len(params[args.inner_rows])):
                    for col in range(len(params[args.inner_cols])):
                        slic_kwargs = {
                            args.outer_cols : params[args.outer_cols][outer_col],
                            args.outer_rows : params[args.outer_rows][outer_row],
                            args.inner_rows : params[args.inner_rows][row],
                            args.inner_cols : params[args.inner_cols][col]
                        }
                        plot_row = row+outer_row*len(params[args.inner_rows])+1
                        plot_col = col+outer_col*len(params[args.inner_cols])
                        segments = slic(image, max_iter=10, spacing=None,
                                multichannel=True, convert2lab=True,
                                enforce_connectivity=True, max_size_factor=3,
                                **slic_kwargs)
                                #, slic_zero=True)

                        ax[plot_row][plot_col].imshow(
                                mark_boundaries(image, segments))
                        #ax[row+1][col+outer_col*len(num_segments)].axis('off')
                        # Remove the tick labels and the ticks
                        ax[plot_row][plot_col].set(xticklabels=[])
                        ax[plot_row][plot_col].set(yticklabels=[])
                        ax[plot_row][plot_col].tick_params(left=False)
                        ax[plot_row][plot_col].tick_params(bottom=False)

                        if row == 0 and outer_row == 0: # and col == 0:
                            ax[plot_row][plot_col].set_title(
                                    args.outer_cols + ': ' +
                                    str(params[args.outer_cols][outer_col]) +
                                    '\n' + args.inner_cols + ': ' +
                                    str(params[args.inner_cols][col]))
                        if col == 0 and outer_col == 0:# and row == 0:
                            ax[plot_row][plot_col].set_ylabel(
                                    args.outer_rows + ': ' +
                                    str(params[args.outer_rows][outer_row]) +
                                    '\n' + args.inner_rows + ': ' +
                                    str(params[args.inner_rows][row]))
        plt.tight_layout()
        image_name = os.path.basename(image_path)
        trajectory_subpath = '/'.join(os.path.dirname(image_path).split('/')[-2:])
        trajectory_outpath = os.path.join(args.outpath, trajectory_subpath)
        if args.dry_run:
            plt.savefig(os.path.join(args.outpath, 'sample_superpixel.png'))
            #plt.show()
            plt.close()
            break
        else:
            plt.savefig(os.path.join(trajectory_outpath, image_name))
        plt.close()

