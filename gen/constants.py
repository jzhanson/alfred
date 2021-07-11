from collections import OrderedDict

########################################################################################################################
# General Settings

DEBUG = True
EVAL = False
LOG_FILE = 'logs_gen'

RECORD_VIDEO_IMAGES = True
RECORD_SMOOTHING_FACTOR = 1
DATA_SAVE_PATH = "dataset/new_trajectories"

OPEN_LOOP = True
FULL_OBSERVABLE_STATE = True

########################################################################################################################
# Generation Ablations

MAX_NUM_OF_OBJ_INSTANCES = 3     # when randomly initializing the scene, create duplicate instance up to this number
PICKUP_REPEAT_MAX = 4            # how many of the target pickup object to generate in [1, MAX] (randomly chosen)
RECEPTACLE_SPARSE_POINTS = 50    # increment for how many points to leave free for sparsely populated receptacles
RECEPTACLE_EMPTY_POINTS = 200    # increment for how many points to leave free for empty receptacles

MIN_VISIBLE_RATIO = 0.0011       # minimum area ratio (with respect to image size) of visible object
PLANNER_MAX_STEPS = 100          # if the generated plan is more than these steps, discard the traj
MAX_EPISODE_LENGTH = 1000        # maximum number of API steps allowed per trajectory

FORCED_SAMPLING = False          # set True for debugging instead of proper sampling
PRUNE_UNREACHABLE_POINTS = True  # prune navigation points that were deemed unreachable by the proprocessing script

########################################################################################################################
# Goals

GOALS = ["pick_and_place_simple",
         "pick_two_obj_and_place",
         "look_at_obj_in_light",
         "pick_clean_then_place_in_recep",
         "pick_heat_then_place_in_recep",
         "pick_cool_then_place_in_recep",
         "pick_and_place_with_movable_recep"]

GOALS_VALID = {"pick_and_place_simple": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
               "pick_two_obj_and_place": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
               "look_at_obj_in_light": {"LivingRoom", "Bedroom"},
               "pick_clean_then_place_in_recep": {"Kitchen", "Bathroom"},
               "pick_heat_then_place_in_recep": {"Kitchen"},
               "pick_cool_then_place_in_recep": {"Kitchen"},
               "pick_and_place_with_movable_recep": {"Kitchen", "LivingRoom", "Bedroom"}}

pddl_goal_type = "pick_and_place_simple"  # default goal type

########################################################################################################################
# Video Settings

# filler frame IDs
BEFORE = 0
MIDDLE = 1
AFTER = 2

# number of image frames to save before and after executing the specified action
SAVE_FRAME_BEFORE_AND_AFTER_COUNTS = {
    'OpenObject': [2, 0, 2],
    'CloseObject': [2, 0, 2],
    'PickupObject': [5, 0, 10],
    'PutObject': [5, 0, 10],
    'CleanObject': [3, 0, 5],
    'HeatObject': [3, 0, 5],
    'CoolObject': [3, 30, 5],
    'ToggleObjectOn': [3, 0, 15],
    'ToggleObjectOff': [1, 0, 5],
    'SliceObject': [3, 0, 7]
}

# FPS
VIDEO_FRAME_RATE = 3

MASK_HIGHLIGHT_COLOR = [255, 255, 0] # Bright yellow
MASK_HIGHLIGHT_OPACITY = 0.30
MASK_BOUNDING_BOX_COLOR = [255, 0, 255] # Bright purple
TEXT_COLOR = [0, 255, 255] # Bright cyan
CHARS_PER_LINE = 40

########################################################################################################################
# Data & Storage

save_path = DATA_SAVE_PATH
data_dict = OrderedDict()  # dictionary for storing trajectory data to be dumped

########################################################################################################################
# Unity Hyperparameters

BUILD_PATH = None
X_DISPLAY = '0'

AGENT_STEP_SIZE = 0.25
AGENT_HORIZON_ADJ = 15
AGENT_ROTATE_ADJ = 90
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 1.5
HORIZON_GRANULARITY = 15

RENDER_IMAGE = True
RENDER_DEPTH_IMAGE = True
RENDER_CLASS_IMAGE = True
RENDER_OBJECT_IMAGE = True

MAX_DEPTH = 5000
STEPS_AHEAD = 5
SCENE_PADDING = STEPS_AHEAD * 3
SCREEN_WIDTH = DETECTION_SCREEN_WIDTH = 300
SCREEN_HEIGHT = DETECTION_SCREEN_HEIGHT = 300
MIN_VISIBLE_PIXELS = 10

# (400) / (600*600) ~ 0.13% area of image
# int(MIN_VISIBLE_RATIO * float(DETECTION_SCREEN_WIDTH) * float(DETECTION_SCREEN_HEIGHT))
# MIN_VISIBLE_PIXELS = int(MIN_VISIBLE_RATIO * float(DETECTION_SCREEN_WIDTH) * float(
#    DETECTION_SCREEN_HEIGHT))  # (400) / (600*600) ~ 0.13% area of image

########################################################################################################################
# Scenes and Objects

TRAIN_SCENE_NUMBERS = list(range(7, 31))           # Train Kitchens (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(207, 231)))  # Train Living Rooms (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(307, 331)))  # Train Bedrooms (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(407, 431)))  # Train Bathrooms (24/30)

TEST_SCENE_NUMBERS = list(range(1, 7))             # Test Kitchens (6/30)
TEST_SCENE_NUMBERS.extend(list(range(201, 207)))   # Test Living Rooms (6/30)
TEST_SCENE_NUMBERS.extend(list(range(301, 307)))   # Test Bedrooms (6/30)
TEST_SCENE_NUMBERS.extend(list(range(401, 407)))   # Test Bathrooms (6/30)

SCENE_NUMBERS = TRAIN_SCENE_NUMBERS + TEST_SCENE_NUMBERS

# Available scenes are [1, 30], [201, 230], [301, 330], and [401, 430]
# Tragically this is hardcoded in ai2thor 2.1.0 in
# ai2thor/controller.py line 429
# I got these splits out of the last number in the first directory of each
# train, valid_seen and valid_unseen task
DATASET_TRAIN_SCENE_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 201, 202, 203, 204, 205,
206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 220, 221, 222, 223,
224, 225, 227, 228, 229, 230, 301, 302, 303, 304, 305, 306, 307, 309, 310, 311,
312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 326, 327, 328, 329,
330, 401, 402, 403, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
417, 418, 419, 420, 421, 422, 423, 426, 427, 428, 429, 430]
DATASET_VALID_SEEN_SCENE_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15,
16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 201, 202, 203, 204,
205, 206, 207, 212, 213, 214, 216, 218, 222, 223, 224, 225, 227, 229, 230, 301,
302, 303, 304, 305, 309, 310, 311, 313, 314, 316, 318, 320, 323, 324, 326, 327,
328, 329, 330, 401, 402, 403, 405, 406, 407, 408, 409, 410, 412, 413, 414, 415,
417, 418, 419, 422, 423, 426, 427, 428, 429]
DATASET_VALID_UNSEEN_SCENE_NUMBERS = [10, 219, 308, 424]
DATASET_TEST_SCENE_NUMBERS = [9, 29, 215, 226, 315, 325, 404, 425]

KITCHEN_TRAIN_SCENE_NUMBERS = list(range(7, 31))
LIVING_ROOM_TRAIN_SCENE_NUMBERS = list(range(207, 231))
BEDROOM_TRAIN_SCENE_NUMBERS = list(range(307, 331))
BATHROOM_TRAIN_SCENE_NUMBERS = list(range(407, 431))

KITCHEN_VALID_SCENE_NUMBERS = list(range(1, 4))
LIVING_ROOM_VALID_SCENE_NUMBERS = list(range(201, 204))
BEDROOM_VALID_SCENE_NUMBERS = list(range(301, 304))
BATHROOM_VALID_SCENE_NUMBERS = list(range(401, 404))

KITCHEN_TEST_SCENE_NUMBERS = list(range(4, 7))
LIVING_ROOM_TEST_SCENE_NUMBERS = list(range(204, 207))
BEDROOM_TEST_SCENE_NUMBERS = list(range(304, 307))
BATHROOM_TEST_SCENE_NUMBERS = list(range(404, 407))

# Scene types.
SCENE_TYPE = {"Kitchen": range(1, 31),
              "LivingRoom": range(201, 231),
              "Bedroom": range(301, 331),
              "Bathroom": range(401, 431)}

OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

OBJECTS_LOWER_TO_UPPER = {obj.lower(): obj for obj in OBJECTS}

OBJECTS_SINGULAR = [
    'alarmclock',
    'apple',
    'armchair',
    'baseballbat',
    'basketball',
    'bathtub',
    'bathtubbasin',
    'bed',
    'blinds',
    'book',
    'boots',
    'bowl',
    'box',
    'bread',
    'butterknife',
    'cabinet',
    'candle',
    'cart',
    'cd',
    'cellphone',
    'chair',
    'cloth',
    'coffeemachine',
    'countertop',
    'creditcard',
    'cup',
    'curtains',
    'desk',
    'desklamp',
    'dishsponge',
    'drawer',
    'dresser',
    'egg',
    'floorlamp',
    'footstool',
    'fork',
    'fridge',
    'garbagecan',
    'glassbottle',
    'handtowel',
    'handtowelholder',
    'houseplant',
    'kettle',
    'keychain',
    'knife',
    'ladle',
    'laptop',
    'laundryhamper',
    'laundryhamperlid',
    'lettuce',
    'lightswitch',
    'microwave',
    'mirror',
    'mug',
    'newspaper',
    'ottoman',
    'painting',
    'pan',
    'papertowel',
    'papertowelroll',
    'pen',
    'pencil',
    'peppershaker',
    'pillow',
    'plate',
    'plunger',
    'poster',
    'pot',
    'potato',
    'remotecontrol',
    'safe',
    'saltshaker',
    'scrubbrush',
    'shelf',
    'showerdoor',
    'showerglass',
    'sink',
    'sinkbasin',
    'soapbar',
    'soapbottle',
    'sofa',
    'spatula',
    'spoon',
    'spraybottle',
    'statue',
    'stoveburner',
    'stoveknob',
    'diningtable',
    'coffeetable',
    'sidetable'
    'teddybear',
    'television',
    'tennisracket',
    'tissuebox',
    'toaster',
    'toilet',
    'toiletpaper',
    'toiletpaperhanger',
    'toiletpaperroll',
    'tomato',
    'towel',
    'towelholder',
    'tvstand',
    'vase',
    'watch',
    'wateringcan',
    'window',
    'winebottle',
]

OBJECTS_PLURAL = [
    'alarmclocks',
    'apples',
    'armchairs',
    'baseballbats',
    'basketballs',
    'bathtubs',
    'bathtubbasins',
    'beds',
    'blinds',
    'books',
    'boots',
    'bottles',
    'bowls',
    'boxes',
    'bread',
    'butterknives',
    'cabinets',
    'candles',
    'carts',
    'cds',
    'cellphones',
    'chairs',
    'cloths',
    'coffeemachines',
    'countertops',
    'creditcards',
    'cups',
    'curtains',
    'desks',
    'desklamps',
    'dishsponges',
    'drawers',
    'dressers',
    'eggs',
    'floorlamps',
    'footstools',
    'forks',
    'fridges',
    'garbagecans',
    'glassbottles',
    'handtowels',
    'handtowelholders',
    'houseplants',
    'kettles',
    'keychains',
    'knives',
    'ladles',
    'laptops',
    'laundryhampers',
    'laundryhamperlids',
    'lettuces',
    'lightswitches',
    'microwaves',
    'mirrors',
    'mugs',
    'newspapers',
    'ottomans',
    'paintings',
    'pans',
    'papertowels',
    'papertowelrolls',
    'pens',
    'pencils',
    'peppershakers',
    'pillows',
    'plates',
    'plungers',
    'posters',
    'pots',
    'potatoes',
    'remotecontrollers',
    'safes',
    'saltshakers',
    'scrubbrushes',
    'shelves',
    'showerdoors',
    'showerglassess',
    'sinks',
    'sinkbasins',
    'soapbars',
    'soapbottles',
    'sofas',
    'spatulas',
    'spoons',
    'spraybottles',
    'statues',
    'stoveburners',
    'stoveknobs',
    'diningtables',
    'coffeetables',
    'sidetable',
    'teddybears',
    'televisions',
    'tennisrackets',
    'tissueboxes',
    'toasters',
    'toilets',
    'toiletpapers',
    'toiletpaperhangers',
    'toiletpaperrolls',
    'tomatoes',
    'towels',
    'towelholders',
    'tvstands',
    'vases',
    'watches',
    'wateringcans',
    'windows',
    'winebottles',
]

ENV_OBJECTS = [
        'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall',
        'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl',
        'Box', 'Bread', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'Cart',
        'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'CoffeeTable',
        'CounterTop', 'CreditCard', 'Cup', 'Curtains', 'Desk', 'DeskLamp',
        'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet',
        'FloorLamp', 'Footstool', 'Fork', 'Fridge', 'GarbageCan',
        'Glassbottle', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle',
        'KeyChain', 'Knife', 'Ladle', 'Lamp', 'Laptop', 'LaundryHamper',
        'LaundryHamperLid', 'Lettuce', 'LightSwitch', 'Microwave', 'Mirror',
        'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan', 'PaperTowelRoll',
        'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger',
        'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe', 'SaltShaker',
        'ScrubBrush', 'Shelf', 'ShowerCurtain', 'ShowerDoor', 'ShowerGlass',
        'ShowerHead', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar',
        'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue',
        'StoveBurner', 'StoveKnob', 'TVStand', 'TeddyBear', 'Television',
        'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper',
        'ToiletPaperHanger', 'Tomato', 'Towel', 'TowelHolder', 'Vase', 'Watch',
        'WateringCan', 'Window', 'WineBottle'
]

NON_ENV_OBJECTS = ['PaintingHanger', 'PaperTowel', 'ToiletPaperRoll',
        'AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced',
        'PotatoSliced', 'TomatoSliced',
]

ALL_OBJECTS = ENV_OBJECTS + NON_ENV_OBJECTS

MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

MOVABLE_RECEPTACLES_SET = set(MOVABLE_RECEPTACLES)
OBJECTS_SET = set(OBJECTS) | MOVABLE_RECEPTACLES_SET

OBJECT_CLASS_TO_ID = {obj: ii for (ii, obj) in enumerate(OBJECTS)}

RECEPTACLES = {
        'BathtubBasin',
        'Bowl',
        'Cup',
        'Drawer',
        'Mug',
        'Plate',
        'Shelf',
        'SinkBasin',
        'Box',
        'Cabinet',
        'CoffeeMachine',
        'CounterTop',
        'Fridge',
        'GarbageCan',
        'HandTowelHolder',
        'Microwave',
        'PaintingHanger',
        'Pan',
        'Pot',
        'StoveBurner',
        'DiningTable',
        'CoffeeTable',
        'SideTable',
        'ToiletPaperHanger',
        'TowelHolder',
        'Safe',
        'BathtubBasin',
        'ArmChair',
        'Toilet',
        'Sofa',
        'Ottoman',
        'Dresser',
        'LaundryHamper',
        'Desk',
        'Bed',
        'Cart',
        'TVStand',
        'Toaster',
    }

NON_RECEPTACLES = OBJECTS_SET - RECEPTACLES

NUM_RECEPTACLES = len(RECEPTACLES)
NUM_CLASSES = len(OBJECTS)

# For generating questions
QUESTION_OBJECT_CLASS_LIST = [
    'Spoon',
    'Potato',
    'Fork',
    'Plate',
    'Egg',
    'Tomato',
    'Bowl',
    'Lettuce',
    'Apple',
    'Knife',
    'Container',
    'Bread',
    'Mug',
]

VAL_RECEPTACLE_OBJECTS = {
    'Pot': {'Apple',
            'AppleSliced',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced'},
    'Pan': {'Apple',
            'AppleSliced',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced'},
    'Bowl': {'Apple',
             'AppleSliced',
             'ButterKnife',
             'DishSponge',
             'Egg',
             'Fork',
             'Knife',
             'Ladle',
             'Lettuce',
             'LettuceSliced',
             'Potato',
             'PotatoSliced',
             'Spatula',
             'Spoon',
             'Tomato',
             'TomatoSliced',
             'Candle',
             'CD',
             'CellPhone',
             'Cloth',
             'CreditCard',
             'DishSponge',
             'KeyChain',
             'Mug',
             'PaperTowel',
             'Pen',
             'Pencil',
             'RemoteControl',
             'Watch'},
    'CoffeeMachine': {'Mug'},
    'Microwave': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'Bread',
                  'BreadSliced',
                  'Cup',
                  'Egg',
                  'Glassbottle',
                  'Mug',
                  'Plate',
                  'Potato',
                  'PotatoSliced',
                  'Tomato',
                  'TomatoSliced'},
    'StoveBurner': {'Kettle',
                    'Pan',
                    'Pot'},
    'Fridge': {'Apple',
               'AppleSliced',
               'Bowl',
               'Bread',
               'BreadSliced',
               'Cup',
               'Egg',
               'Glassbottle',
               'Lettuce',
               'LettuceSliced',
               'Mug',
               'Pan',
               'Plate',
               'Pot',
               'Potato',
               'PotatoSliced',
               'Tomato',
               'TomatoSliced',
               'WineBottle'},
    'Mug': {'ButterKnife',
            'Fork',
            'Knife',
            'Pen',
            'Pencil',
            'Spoon',
            'KeyChain',
            'Watch'},
    'Plate': {'Apple',
              'AppleSliced',
              'ButterKnife',
              'DishSponge',
              'Egg',
              'Fork',
              'Knife',
              'Ladle',
              'Lettuce',
              'LettuceSliced',
              'Mug',
              'Potato',
              'PotatoSliced',
              'Spatula',
              'Spoon',
              'Tomato',
              'TomatoSliced',
              'AlarmClock',
              'Book',
              'Candle',
              'CD',
              'CellPhone',
              'Cloth',
              'CreditCard',
              'DishSponge',
              'Glassbottle',
              'KeyChain',
              'Mug',
              'PaperTowel',
              'Pen',
              'Pencil',
              'TissueBox',
              'Watch'},
    'Cup': {'ButterKnife',
            'Fork',
            'Spoon'},
    'Sofa': {'BasketBall',
             'Book',
             'Box',
             'CellPhone',
             'Cloth',
             'CreditCard',
             'KeyChain',
             'Laptop',
             'Newspaper',
             'Pillow',
             'RemoteControl'},
    'ArmChair': {'BasketBall',
                 'Book',
                 'Box',
                 'CellPhone',
                 'Cloth',
                 'CreditCard',
                 'KeyChain',
                 'Laptop',
                 'Newspaper',
                 'Pillow',
                 'RemoteControl'},
    'Box': {'AlarmClock',
            'Book',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'DishSponge',
            'Glassbottle',
            'KeyChain',
            'Mug',
            'PaperTowel',
            'Pen',
            'Pencil',
            'RemoteControl',
            'Statue',
            'TissueBox',
            'Vase',
            'Watch'},
    'Ottoman': {'BasketBall',
                'Book',
                'Box',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'KeyChain',
                'Laptop',
                'Newspaper',
                'Pillow',
                'RemoteControl'},
    'Dresser': {'AlarmClock',
                'BasketBall',
                'Book',
                'Bowl',
                'Box',
                'Candle',
                'CD',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'Cup',
                'Glassbottle',
                'KeyChain',
                'Laptop',
                'Mug',
                'Newspaper',
                'Pen',
                'Pencil',
                'Plate',
                'RemoteControl',
                'SprayBottle',
                'Statue',
                'TennisRacket',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Vase',
                'Watch',
                'WateringCan',
                'WineBottle'},
    'LaundryHamper': {'Cloth'},
    'Desk': {'AlarmClock',
             'BasketBall',
             'Book',
             'Bowl',
             'Box',
             'Candle',
             'CD',
             'CellPhone',
             'Cloth',
             'CreditCard',
             'Cup',
             'Glassbottle',
             'KeyChain',
             'Laptop',
             'Mug',
             'Newspaper',
             'Pen',
             'Pencil',
             'Plate',
             'RemoteControl',
             'SoapBottle',
             'SprayBottle',
             'Statue',
             'TennisRacket',
             'TissueBox',
             'ToiletPaper',
             'ToiletPaperRoll',
             'Vase',
             'Watch',
             'WateringCan',
             'WineBottle'},
    'Bed': {'BaseballBat',
            'BasketBall',
            'Book',
            'CellPhone',
            'Laptop',
            'Newspaper',
            'Pillow',
            'TennisRacket'},
    'Toilet': {'Candle',
               'Cloth',
               'DishSponge',
               'Newspaper',
               'PaperTowel',
               'SoapBar',
               'SoapBottle',
               'SprayBottle',
               'TissueBox',
               'ToiletPaper',
               'ToiletPaperRoll',
               'HandTowel'},
    'ToiletPaperHanger': {'ToiletPaper',
                          'ToiletPaperRoll'},
    'TowelHolder': {'Towel'},
    'HandTowelHolder': {'HandTowel'},
    'Cart': {'Candle',
             'Cloth',
             'DishSponge',
             'Mug',
             'PaperTowel',
             'Plunger',
             'SoapBar',
             'SoapBottle',
             'SprayBottle',
             'Statue',
             'TissueBox',
             'ToiletPaper',
             'ToiletPaperRoll',
             'Vase',
             'HandTowel'},
    'BathtubBasin': {'Cloth',
                     'DishSponge',
                     'SoapBar',
                     'HandTowel'},
    'SinkBasin': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'ButterKnife',
                  'Cloth',
                  'Cup',
                  'DishSponge',
                  'Egg',
                  'Glassbottle',
                  'Fork',
                  'Kettle',
                  'Knife',
                  'Ladle',
                  'Lettuce',
                  'LettuceSliced',
                  'Mug',
                  'Pan',
                  'Plate',
                  'Pot',
                  'Potato',
                  'PotatoSliced',
                  'SoapBar',
                  'Spatula',
                  'Spoon',
                  'Tomato',
                  'TomatoSliced',
                  'HandTowel'},
    'Cabinet': {'Book',
                'Bowl',
                'Box',
                'Candle',
                'CD',
                'Cloth',
                'Cup',
                'DishSponge',
                'Glassbottle',
                'Kettle',
                'Ladle',
                'Mug',
                'Newspaper',
                'Pan',
                'PepperShaker',
                'Plate',
                'Plunger',
                'Pot',
                'SaltShaker',
                'SoapBar',
                'SoapBottle',
                'SprayBottle',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Vase',
                'WateringCan',
                'WineBottle',
                'HandTowel'},
    'TableTop': {'AlarmClock',
                 'Apple',
                 'AppleSliced',
                 'BaseballBat',
                 'BasketBall',
                 'Book',
                 'Bowl',
                 'Box',
                 'Bread',
                 'BreadSliced',
                 'ButterKnife',
                 'Candle',
                 'CD',
                 'CellPhone',
                 'Cloth',
                 'CreditCard',
                 'Cup',
                 'DishSponge',
                 'Glassbottle',
                 'Egg',
                 'Fork',
                 'Kettle',
                 'KeyChain',
                 'Knife',
                 'Ladle',
                 'Laptop',
                 'Lettuce',
                 'LettuceSliced',
                 'Mug',
                 'Newspaper',
                 'Pan',
                 'PaperTowel',
                 'Pen',
                 'Pencil',
                 'PepperShaker',
                 'Plate',
                 'Pot',
                 'Potato',
                 'PotatoSliced',
                 'RemoteControl',
                 'SaltShaker',
                 'SoapBar',
                 'SoapBottle',
                 'Spatula',
                 'Spoon',
                 'SprayBottle',
                 'Statue',
                 'TennisRacket',
                 'TissueBox',
                 'ToiletPaper',
                 'ToiletPaperRoll',
                 'Tomato',
                 'TomatoSliced',
                 'Vase',
                 'Watch',
                 'WateringCan',
                 'WineBottle',
                 'HandTowel'},
    'CounterTop': {'AlarmClock',
                   'Apple',
                   'AppleSliced',
                   'BaseballBat',
                   'BasketBall',
                   'Book',
                   'Bowl',
                   'Box',
                   'Bread',
                   'BreadSliced',
                   'ButterKnife',
                   'Candle',
                   'CD',
                   'CellPhone',
                   'Cloth',
                   'CreditCard',
                   'Cup',
                   'DishSponge',
                   'Egg',
                   'Glassbottle',
                   'Fork',
                   'Kettle',
                   'KeyChain',
                   'Knife',
                   'Ladle',
                   'Laptop',
                   'Lettuce',
                   'LettuceSliced',
                   'Mug',
                   'Newspaper',
                   'Pan',
                   'PaperTowel',
                   'Pen',
                   'Pencil',
                   'PepperShaker',
                   'Plate',
                   'Pot',
                   'Potato',
                   'PotatoSliced',
                   'RemoteControl',
                   'SaltShaker',
                   'SoapBar',
                   'SoapBottle',
                   'Spatula',
                   'Spoon',
                   'SprayBottle',
                   'Statue',
                   'TennisRacket',
                   'TissueBox',
                   'ToiletPaper',
                   'ToiletPaperRoll',
                   'Tomato',
                   'TomatoSliced',
                   'Vase',
                   'Watch',
                   'WateringCan',
                   'WineBottle',
                   'HandTowel'},
    'Shelf': {'AlarmClock',
              'Book',
              'Bowl',
              'Box',
              'Candle',
              'CD',
              'CellPhone',
              'Cloth',
              'CreditCard',
              'Cup',
              'DishSponge',
              'Glassbottle',
              'Kettle',
              'KeyChain',
              'Mug',
              'Newspaper',
              'PaperTowel',
              'Pen',
              'Pencil',
              'PepperShaker',
              'Plate',
              'Pot',
              'RemoteControl',
              'SaltShaker',
              'SoapBar',
              'SoapBottle',
              'SprayBottle',
              'Statue',
              'TissueBox',
              'ToiletPaper',
              'ToiletPaperRoll',
              'Vase',
              'Watch',
              'WateringCan',
              'WineBottle',
              'HandTowel'},
    'Drawer': {'Book',
               'ButterKnife',
               'Candle',
               'CD',
               'CellPhone',
               'Cloth',
               'CreditCard',
               'DishSponge',
               'Fork',
               'KeyChain',
               'Knife',
               'Ladle',
               'Newspaper',
               'Pen',
               'Pencil',
               'PepperShaker',
               'RemoteControl',
               'SaltShaker',
               'SoapBar',
               'SoapBottle',
               'Spatula',
               'Spoon',
               'SprayBottle',
               'TissueBox',
               'ToiletPaper',
               'ToiletPaperRoll',
               'Watch',
               'WateringCan',
               'HandTowel'},
    'GarbageCan': {'Apple',
                   'AppleSliced',
                   'Bread',
                   'BreadSliced',
                   'CD',
                   'Cloth',
                   'DishSponge',
                   'Egg',
                   'Lettuce',
                   'LettuceSliced',
                   'Newspaper',
                   'PaperTowel',
                   'Pen',
                   'Pencil',
                   'Potato',
                   'PotatoSliced',
                   'SoapBar',
                   'SoapBottle',
                   'SprayBottle',
                   'TissueBox',
                   'ToiletPaper',
                   'ToiletPaperRoll',
                   'Tomato',
                   'TomatoSliced',
                   'WineBottle',
                   'HandTowel'},
    'Safe': {'CD',
             'CellPhone',
             'CreditCard',
             'KeyChain',
             'Statue',
             'Vase',
             'Watch'},
    'TVStand': {'TissueBox'},
    'Toaster': {'BreadSliced'},
}
VAL_RECEPTACLE_OBJECTS['DiningTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
VAL_RECEPTACLE_OBJECTS['CoffeeTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
VAL_RECEPTACLE_OBJECTS['SideTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
del VAL_RECEPTACLE_OBJECTS['TableTop']

NON_RECEPTACLES_SET = (OBJECTS_SET - set(VAL_RECEPTACLE_OBJECTS.keys())) | set(MOVABLE_RECEPTACLES)

VAL_ACTION_OBJECTS = {
    'Heatable': {'Apple',
                 'AppleSliced',
                 'Bread',
                 'BreadSliced',
                 'Cup',
                 'Egg',
                 'Mug',
                 'Plate',
                 'Potato',
                 'PotatoSliced',
                 'Tomato',
                 'TomatoSliced'},
    'Coolable': {'Apple',
                 'AppleSliced',
                 'Bowl',
                 'Bread',
                 'BreadSliced',
                 'Cup',
                 'Egg',
                 'Lettuce',
                 'LettuceSliced',
                 'Mug',
                 'Pan',
                 'Plate',
                 'Pot',
                 'Potato',
                 'PotatoSliced',
                 'Tomato',
                 'TomatoSliced',
                 'WineBottle'},
    'Cleanable': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'ButterKnife',
                  'Cloth',
                  'Cup',
                  'DishSponge',
                  'Egg',
                  'Fork',
                  'Kettle',
                  'Knife',
                  'Ladle',
                  'Lettuce',
                  'LettuceSliced',
                  'Mug',
                  'Pan',
                  'Plate',
                  'Pot',
                  'Potato',
                  'PotatoSliced',
                  'SoapBar',
                  'Spatula',
                  'Spoon',
                  'Tomato',
                  'TomatoSliced'},
    'Toggleable': {'DeskLamp',
                   'FloorLamp'},
    'Sliceable': {'Apple',
                  'Bread',
                  'Egg',
                  'Lettuce',
                  'Potato',
                  'Tomato'}
}

# object parents
OBJ_PARENTS = {obj: obj for obj in OBJECTS}
OBJ_PARENTS['AppleSliced'] = 'Apple'
OBJ_PARENTS['BreadSliced'] = 'Bread'
OBJ_PARENTS['EggCracked'] = 'Egg'
OBJ_PARENTS['LettuceSliced'] = 'Lettuce'
OBJ_PARENTS['PotatoSliced'] = 'Potato'
OBJ_PARENTS['TomatoSliced'] = 'Tomato'

# Some of the object naming conventions aren't shared, e.g. Bread_0 vs Bread
NUM_SLICED_OBJ_PARTS = {}
NUM_SLICED_OBJ_PARTS['Apple'] = 10
NUM_SLICED_OBJ_PARTS['Bread'] = 10
NUM_SLICED_OBJ_PARTS['Egg'] = 1
NUM_SLICED_OBJ_PARTS['Lettuce'] = 7
NUM_SLICED_OBJ_PARTS['Potato'] = 8
NUM_SLICED_OBJ_PARTS['Tomato'] = 7

# force a different horizon view for objects of (type, location). If the location is None, force this horizon for all
# objects of that type.
FORCED_HORIZON_OBJS = {
    ('FloorLamp', None): 0,
    ('Fridge', 18): 30,
    ('Toilet', None): 15,
}

# openable objects with fixed states for transport.
FORCED_OPEN_STATE_ON_PICKUP = {
    'Laptop': False,
}

# list of openable classes.
OPENABLE_CLASS_LIST = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']
OPENABLE_CLASS_SET = set(OPENABLE_CLASS_LIST)

########################################################################################################################
# Interaction Exploration actions

ACTIONS_INTERACT = 'Interact'
NAV_ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']
INT_ACTIONS = ['OpenObject', 'CloseObject', 'PickupObject', 'PutObject',
        'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']
SIMPLE_ACTIONS = NAV_ACTIONS + [ACTIONS_INTERACT]
COMPLEX_ACTIONS = NAV_ACTIONS + INT_ACTIONS
INDEX_TO_ACTION_SIMPLE = dict(enumerate(SIMPLE_ACTIONS))
ACTION_TO_INDEX_SIMPLE = dict((v,k) for k,v in INDEX_TO_ACTION_SIMPLE.items())
INDEX_TO_ACTION_COMPLEX = dict(enumerate(COMPLEX_ACTIONS))
ACTION_TO_INDEX_COMPLEX = dict((v,k) for k,v in
        INDEX_TO_ACTION_COMPLEX.items())

KEY_TO_ACTION = {
        ' ' : 'MoveAhead',
        'w' : 'MoveAhead',
        'a' : 'RotateLeft',
        'd' : 'RotateLeft',
        'h' : 'RotateLeft',
        'j' : 'LookDown',
        'k' : 'LookUp',
        'l' : 'RotateRight',
        'ma' : 'MoveAhead',
        'rl' : 'RotateLeft',
        'rr' : 'RotateRight',
        'ld' : 'LookDown',
        'lu' : 'LookUp',
        'i' : 'Interact',
        'oo' : 'OpenObject',
        'co' : 'CloseObject',
        'pio' : 'PickupObject',
        'puo' : 'PutObject',
        'toon' : 'ToggleObjectOn',
        'toof' : 'ToggleObjectOff',
        'so' : 'SliceObject',
        }

# Interaction exploration total coverages
COVERAGE_TYPES = ['navigation', 'navigation_pose', 'interaction_by_object',
        'state_change_by_object', 'interaction_by_type',
        'state_change_by_type']

# See gen/scripts/calculate_scene_coverages.py
# Yet another note: coverage can behave strangely for the single_interact case.
SCENE_NAVIGATION_COVERAGES = {1: 129, 2: 115, 3: 104, 4: 69, 5: 99, 6: 134, 7:
        268, 8: 172, 9: 76, 10: 203, 11: 67, 12: 79, 13: 176, 14: 118, 15: 96,
        16: 198, 17: 70, 18: 222, 19: 70, 20: 80, 21: 105, 22: 136, 23: 96, 24:
        61, 25: 27, 26: 76, 27: 43, 28: 101, 29: 65, 30: 59, 301: 87, 302: 45,
        303: 81, 304: 109, 305: 85, 306: 93, 307: 89, 308: 98, 309: 362, 310:
        67, 311: 225, 312: 93, 313: 49, 314: 80, 315: 101, 316: 58, 317: 127,
        318: 88, 319: 88, 320: 60, 321: 82, 322: 98, 323: 189, 324: 93, 325:
        186, 326: 101, 327: 77, 328: 62, 329: 88, 330: 136, 401: 98, 402: 94,
        403: 64, 404: 50, 405: 27, 406: 98, 407: 33, 408: 37, 409: 39, 410: 80,
        411: 67, 412: 49, 413: 69, 414: 50, 415: 57, 416: 70, 417: 58, 418: 48,
        419: 31, 420: 28, 421: 31, 422: 40, 423: 56, 424: 38, 425: 25, 426: 50,
        427: 51, 428: 55, 429: 63, 430: 106, 201: 189, 202: 143, 203: 498, 204:
        214, 205: 248, 206: 115, 207: 145, 208: 213, 209: 304, 210: 268, 211:
        125, 212: 104, 213: 292, 214: 186, 215: 377, 216: 150, 217: 153, 218:
        420, 219: 213, 220: 237, 221: 118, 222: 106, 223: 219, 224: 273, 225:
        154, 226: 79, 227: 199, 228: 160, 229: 208, 230: 359}
SCENE_INTERACTION_COVERAGES_BY_OBJECT_NO_SLICES = {1: 128, 2: 128, 3: 100, 4:
        88, 5: 132, 6: 114, 7: 122, 8: 132, 9: 152, 10: 104, 11: 108, 12: 116,
        13: 152, 14: 88, 15: 106, 16: 156, 17: 136, 18: 122, 19: 122, 20: 102,
        21: 82, 22: 122, 23: 104, 24: 128, 25: 108, 26: 96, 27: 108, 28: 100,
        29: 92, 30: 162, 301: 88, 302: 48, 303: 60, 304: 54, 305: 58, 306: 48,
        307: 58, 308: 60, 309: 66, 310: 56, 311: 52, 312: 38, 313: 60, 314: 50,
        315: 66, 316: 46, 317: 76, 318: 72, 319: 66, 320: 38, 321: 44, 322: 64,
        323: 70, 324: 64, 325: 90, 326: 62, 327: 62, 328: 48, 329: 52, 330: 80,
        401: 40, 402: 48, 403: 50, 404: 38, 405: 44, 406: 38, 407: 48, 408: 44,
        409: 46, 410: 50, 411: 44, 412: 44, 413: 54, 414: 56, 415: 42, 416: 40,
        417: 44, 418: 40, 419: 42, 420: 36, 421: 54, 422: 50, 423: 56, 424: 44,
        425: 46, 426: 52, 427: 56, 428: 44, 429: 36, 430: 50, 201: 52, 202: 32,
        203: 64, 204: 60, 205: 40, 206: 40, 207: 34, 208: 34, 209: 54, 210: 50,
        211: 40, 212: 42, 213: 56, 214: 42, 215: 60, 216: 48, 217: 48, 218: 46,
        219: 64, 220: 46, 221: 34, 222: 44, 223: 40, 224: 88, 225: 40, 226: 52,
        227: 84, 228: 36, 229: 54, 230: 54}
SCENE_INTERACTION_COVERAGES_BY_OBJECT = {1: 214, 2: 214, 3: 186, 4: 174, 5:
        218, 6: 200, 7: 208, 8: 218, 9: 238, 10: 190, 11: 194, 12: 202, 13:
        238, 14: 174, 15: 192, 16: 242, 17: 222, 18: 208, 19: 208, 20: 188, 21:
        168, 22: 208, 23: 190, 24: 214, 25: 194, 26: 182, 27: 194, 28: 186, 29:
        178, 30: 248, 301: 88, 302: 48, 303: 60, 304: 54, 305: 58, 306: 48,
        307: 58, 308: 60, 309: 66, 310: 56, 311: 52, 312: 38, 313: 60, 314: 50,
        315: 66, 316: 46, 317: 76, 318: 72, 319: 66, 320: 38, 321: 44, 322: 64,
        323: 70, 324: 64, 325: 90, 326: 62, 327: 62, 328: 48, 329: 52, 330: 80,
        401: 40, 402: 48, 403: 50, 404: 38, 405: 44, 406: 38, 407: 48, 408: 44,
        409: 46, 410: 50, 411: 44, 412: 44, 413: 54, 414: 56, 415: 42, 416: 40,
        417: 44, 418: 40, 419: 42, 420: 36, 421: 54, 422: 50, 423: 56, 424: 44,
        425: 46, 426: 52, 427: 56, 428: 44, 429: 36, 430: 50, 201: 52, 202: 32,
        203: 64, 204: 60, 205: 40, 206: 40, 207: 34, 208: 34, 209: 54, 210: 50,
        211: 40, 212: 42, 213: 56, 214: 42, 215: 60, 216: 48, 217: 48, 218: 46,
        219: 64, 220: 46, 221: 34, 222: 44, 223: 40, 224: 88, 225: 40, 226: 52,
        227: 84, 228: 36, 229: 54, 230: 54}
SCENE_STATE_CHANGE_COVERAGES_BY_OBJECT_NO_SLICES = {1: 24, 2: 21, 3: 22, 4: 21,
        5: 21, 6: 21, 7: 22, 8: 23, 9: 21, 10: 23, 11: 21, 12: 21, 13: 22, 14:
        21, 15: 22, 16: 24, 17: 24, 18: 21, 19: 21, 20: 22, 21: 21, 22: 21, 23:
        24, 24: 21, 25: 21, 26: 21, 27: 22, 28: 21, 29: 21, 30: 23, 301: 0,
        302: 0, 303: 0, 304: 0, 305: 0, 306: 0, 307: 0, 308: 0, 309: 0, 310: 0,
        311: 0, 312: 0, 313: 0, 314: 0, 315: 0, 316: 0, 317: 0, 318: 0, 319: 0,
        320: 0, 321: 0, 322: 0, 323: 0, 324: 0, 325: 0, 326: 0, 327: 0, 328: 0,
        329: 0, 330: 0, 401: 0, 402: 0, 403: 0, 404: 0, 405: 0, 406: 0, 407: 0,
        408: 0, 409: 0, 410: 0, 411: 0, 412: 0, 413: 0, 414: 0, 415: 0, 416: 0,
        417: 0, 418: 0, 419: 0, 420: 0, 421: 0, 422: 0, 423: 0, 424: 0, 425: 0,
        426: 0, 427: 0, 428: 0, 429: 0, 430: 0, 201: 0, 202: 0, 203: 0, 204: 0,
        205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0,
        214: 0, 215: 0, 216: 0, 217: 0, 218: 0, 219: 0, 220: 0, 221: 0, 222: 0,
        223: 0, 224: 0, 225: 0, 226: 0, 227: 0, 228: 0, 229: 0, 230: 0}
SCENE_STATE_CHANGE_COVERAGES_BY_OBJECT = {1: 152, 2: 149, 3: 148, 4: 148, 5:
        149, 6: 148, 7: 149, 8: 152, 9: 148, 10: 150, 11: 148, 12: 147, 13:
        148, 14: 148, 15: 149, 16: 152, 17: 152, 18: 148, 19: 147, 20: 149, 21:
        147, 22: 148, 23: 151, 24: 147, 25: 148, 26: 147, 27: 150, 28: 147, 29:
        148, 30: 152, 301: 0, 302: 0, 303: 0, 304: 0, 305: 0, 306: 0, 307: 0,
        308: 0, 309: 0, 310: 0, 311: 0, 312: 0, 313: 0, 314: 0, 315: 0, 316: 0,
        317: 0, 318: 0, 319: 0, 320: 0, 321: 0, 322: 0, 323: 0, 324: 0, 325: 0,
        326: 0, 327: 0, 328: 0, 329: 0, 330: 0, 401: 4, 402: 3, 403: 4, 404: 3,
        405: 3, 406: 3, 407: 3, 408: 3, 409: 3, 410: 3, 411: 3, 412: 0, 413: 3,
        414: 4, 415: 3, 416: 3, 417: 0, 418: 3, 419: 3, 420: 3, 421: 4, 422: 3,
        423: 3, 424: 3, 425: 3, 426: 3, 427: 4, 428: 3, 429: 3, 430: 4, 201: 0,
        202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0,
        211: 0, 212: 0, 213: 0, 214: 0, 215: 0, 216: 0, 217: 0, 218: 0, 219: 0,
        220: 0, 221: 0, 222: 0, 223: 0, 224: 0, 225: 0, 226: 0, 227: 0, 228: 0,
        229: 0, 230: 0}
SCENE_INTERACTION_COVERAGES_BY_TYPE_NO_SLICES = {1: 88, 2: 78, 3: 74, 4: 70, 5:
        80, 6: 74, 7: 82, 8: 80, 9: 72, 10: 82, 11: 74, 12: 68, 13: 74, 14: 68,
        15: 76, 16: 76, 17: 82, 18: 78, 19: 68, 20: 72, 21: 70, 22: 74, 23: 78,
        24: 70, 25: 70, 26: 68, 27: 72, 28: 70, 29: 70, 30: 80, 301: 56, 302:
        44, 303: 50, 304: 50, 305: 50, 306: 40, 307: 48, 308: 48, 309: 50, 310:
        48, 311: 46, 312: 36, 313: 50, 314: 42, 315: 42, 316: 44, 317: 50, 318:
        46, 319: 44, 320: 38, 321: 38, 322: 42, 323: 44, 324: 42, 325: 40, 326:
        52, 327: 44, 328: 44, 329: 44, 330: 44, 401: 36, 402: 36, 403: 36, 404:
        32, 405: 32, 406: 30, 407: 34, 408: 32, 409: 32, 410: 34, 411: 32, 412:
        34, 413: 36, 414: 36, 415: 34, 416: 32, 417: 34, 418: 32, 419: 34, 420:
        32, 421: 36, 422: 36, 423: 36, 424: 32, 425: 36, 426: 34, 427: 38, 428:
        34, 429: 32, 430: 40, 201: 50, 202: 32, 203: 52, 204: 40, 205: 34, 206:
        30, 207: 30, 208: 30, 209: 42, 210: 36, 211: 38, 212: 40, 213: 34, 214:
        36, 215: 38, 216: 36, 217: 36, 218: 40, 219: 46, 220: 36, 221: 34, 222:
        32, 223: 36, 224: 46, 225: 38, 226: 34, 227: 36, 228: 36, 229: 46, 230:
        42}
SCENE_STATE_CHANGE_COVERAGES_BY_TYPE_NO_SLICES = {1: 43, 2: 40, 3: 39, 4: 39,
        5: 40, 6: 39, 7: 40, 8: 43, 9: 39, 10: 41, 11: 39, 12: 38, 13: 39, 14:
        39, 15: 40, 16: 43, 17: 43, 18: 39, 19: 38, 20: 40, 21: 38, 22: 39, 23:
        42, 24: 38, 25: 39, 26: 38, 27: 40, 28: 38, 29: 39, 30: 43, 301: 0,
        302: 0, 303: 0, 304: 0, 305: 0, 306: 0, 307: 0, 308: 0, 309: 0, 310: 0,
        311: 0, 312: 0, 313: 0, 314: 0, 315: 0, 316: 0, 317: 0, 318: 0, 319: 0,
        320: 0, 321: 0, 322: 0, 323: 0, 324: 0, 325: 0, 326: 0, 327: 0, 328: 0,
        329: 0, 330: 0, 401: 4, 402: 3, 403: 4, 404: 3, 405: 3, 406: 3, 407: 3,
        408: 3, 409: 3, 410: 3, 411: 3, 412: 0, 413: 3, 414: 4, 415: 3, 416: 3,
        417: 0, 418: 3, 419: 3, 420: 3, 421: 4, 422: 3, 423: 3, 424: 3, 425: 3,
        426: 3, 427: 4, 428: 3, 429: 3, 430: 4, 201: 0, 202: 0, 203: 0, 204: 0,
        205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0,
        214: 0, 215: 0, 216: 0, 217: 0, 218: 0, 219: 0, 220: 0, 221: 0, 222: 0,
        223: 0, 224: 0, 225: 0, 226: 0, 227: 0, 228: 0, 229: 0, 230: 0}
SCENE_INTERACTION_COVERAGES_BY_TYPE = {1: 100, 2: 90, 3: 86, 4: 82, 5: 92, 6:
        86, 7: 94, 8: 92, 9: 84, 10: 94, 11: 86, 12: 80, 13: 86, 14: 80, 15:
        88, 16: 88, 17: 94, 18: 90, 19: 80, 20: 84, 21: 82, 22: 86, 23: 90, 24:
        82, 25: 82, 26: 80, 27: 84, 28: 82, 29: 82, 30: 92, 301: 56, 302: 44,
        303: 50, 304: 50, 305: 50, 306: 40, 307: 48, 308: 48, 309: 50, 310: 48,
        311: 46, 312: 36, 313: 50, 314: 42, 315: 42, 316: 44, 317: 50, 318: 46,
        319: 44, 320: 38, 321: 38, 322: 42, 323: 44, 324: 42, 325: 40, 326: 52,
        327: 44, 328: 44, 329: 44, 330: 44, 401: 36, 402: 36, 403: 36, 404: 32,
        405: 32, 406: 30, 407: 34, 408: 32, 409: 32, 410: 34, 411: 32, 412: 34,
        413: 36, 414: 36, 415: 34, 416: 32, 417: 34, 418: 32, 419: 34, 420: 32,
        421: 36, 422: 36, 423: 36, 424: 32, 425: 36, 426: 34, 427: 38, 428: 34,
        429: 32, 430: 40, 201: 50, 202: 32, 203: 52, 204: 40, 205: 34, 206: 30,
        207: 30, 208: 30, 209: 42, 210: 36, 211: 38, 212: 40, 213: 34, 214: 36,
        215: 38, 216: 36, 217: 36, 218: 40, 219: 46, 220: 36, 221: 34, 222: 32,
        223: 36, 224: 46, 225: 38, 226: 34, 227: 36, 228: 36, 229: 46, 230: 42}
SCENE_STATE_CHANGE_COVERAGES_BY_TYPE = {1: 56, 2: 53, 3: 52, 4: 52, 5: 53, 6:
        52, 7: 53, 8: 56, 9: 52, 10: 54, 11: 52, 12: 51, 13: 52, 14: 52, 15:
        53, 16: 56, 17: 56, 18: 52, 19: 51, 20: 53, 21: 51, 22: 52, 23: 55, 24:
        51, 25: 52, 26: 51, 27: 53, 28: 51, 29: 52, 30: 56, 301: 0, 302: 0,
        303: 0, 304: 0, 305: 0, 306: 0, 307: 0, 308: 0, 309: 0, 310: 0, 311: 0,
        312: 0, 313: 0, 314: 0, 315: 0, 316: 0, 317: 0, 318: 0, 319: 0, 320: 0,
        321: 0, 322: 0, 323: 0, 324: 0, 325: 0, 326: 0, 327: 0, 328: 0, 329: 0,
        330: 0, 401: 4, 402: 3, 403: 4, 404: 3, 405: 3, 406: 3, 407: 3, 408: 3,
        409: 3, 410: 3, 411: 3, 412: 0, 413: 3, 414: 4, 415: 3, 416: 3, 417: 0,
        418: 3, 419: 3, 420: 3, 421: 4, 422: 3, 423: 3, 424: 3, 425: 3, 426: 3,
        427: 4, 428: 3, 429: 3, 430: 4, 201: 0, 202: 0, 203: 0, 204: 0, 205: 0,
        206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0,
        215: 0, 216: 0, 217: 0, 218: 0, 219: 0, 220: 0, 221: 0, 222: 0, 223: 0,
        224: 0, 225: 0, 226: 0, 227: 0, 228: 0, 229: 0, 230: 0}
########################################################################################################################
