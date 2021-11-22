from pygit2 import Repository

# ****** Modi ******
MODE_REAL_INTENSE = 0
MODE_DEBUG = 1
MODE_REAL = 2
MODE_NO_SHUFFLE = 3

DEBUG_MODE = MODE_REAL
DEBUG_MODE = MODE_REAL_INTENSE
#DEBUG_MODE = MODE_DEBUG
#DEBUG_MODE = MODE_NO_SHUFFLE

# ****** Evaluation ******
PREDICT_FUTURE = True
THRESHOLD_MSE_PER_CROP = 0.115 if PREDICT_FUTURE else 0.78  # the mean squared error for one patch
THRESHOLD_PER_PIXEL = 0.4
THRESHOLD_VOTES_PER_IMAGE = 20   # the amount of votes you want to have for a significant change per picture
                                 # meaning every crop with a threshold > MSE_PER_CROP votes 1

USE_GC = True

AMOUNT_OF_GLOBAL_CONTEXTS = 16 if USE_GC else 2  # how many global context will be used to make predictions during evaluation

VOLCANOES_OF_INTEREST = {
    #"fuego": (20180525, 20180630), # only has shifted images
    #"kilauea": (20180401, 20180801), # pictures too big
    #"krakatau": (20180601, 20190201), # too many examples

    #"krakatau": (20180901, 20190201),  # usually the one above, but then we have to many examples...
    #"pitonfournaise": (20180501, 20180630),
    #"pitonfournaise": (20180501, 20180630),
    #"pitonfournaise": (20180901, 20181130),
    #"pitonfournaise": (20190201, 20190415),
    #"ambrym": (20181201, 20190201),

    #"ambrym": (20181213, 20181219), # something happens here!
    #"pitonfournaise": (20190201, 20190415), # a false positive
}

if MODE_DEBUG == DEBUG_MODE:
    PERIOD_EVAL_START = 20180613
    PERIOD_EVAL_END = 20181011

# ****** Paths ******
ROOT = 'D:/trunk/sar_lui_net/'

TIF_PATH = 'F:/big_data/sar_images/data_sar_geotiff'
ALTERNATIVE_TIF_PATH = ROOT + 'raw_sar_data/'

BRANCH_NAME = Repository(ROOT).head.shorthand

DATABASE_DIR = ROOT + 'database/'
TRAIN_DATABASE = DATABASE_DIR + 'training.tfrecord'
TEST_DATABASE = DATABASE_DIR + 'test.tfrecord'
EVAL_DATABASE = DATABASE_DIR + 'evaluation.tfrecord'
EVAL_FILE_PAIRS = DATABASE_DIR + 'evalfiles.json'
TRAIN_FILE_PAIRS = DATABASE_DIR + 'trainfiles.json'
TEST_FILE_PAIRS = DATABASE_DIR + 'testfiles.json'
CSV_CLEANED_FILES = DATABASE_DIR + 'cleaned_data.csv'

BRANCH_DIR = ROOT + 'branches/' + BRANCH_NAME + '/'
PREDICTION_DIR = BRANCH_DIR + 'predictions/'

MODEL_DIR = BRANCH_DIR + 'models/'
MODEL_DIR_INBETWEEN_PATH = MODEL_DIR + 'inbetween/'
MODEL_DIR_INBETWEEN_FILE = MODEL_DIR_INBETWEEN_PATH + 'saved-model-{epoch:02d}-{loss:.3f}.h5'
MODEL_DIR_SAVE = MODEL_DIR + 'complete_model.h5'
TRAINING_HISTORY_DIR = MODEL_DIR + 'history.json'

# ****** Trainig ******
TRAIN_EXAMPLE_COUNT = 134908
TEST_EXAMPLE_COUNT = 0

BATCH_SIZE = 32
EPOCHS = 100
EPOCH_STEP_REGULATOR = 3  # only go through 1/REGULATOR of all samples

LEARNING_RATE = 0.5
RHO = 0.98

if DEBUG_MODE == MODE_REAL_INTENSE:
    LEARNING_RATE = 0.5
    RHO = 0.99
    EPOCHS = 1000

if DEBUG_MODE == MODE_DEBUG:
    RHO = 0.95
    LEARNING_RATE = 1
    EPOCH_STEP_REGULATOR = 20


# Image
GLOBAL_CONTEXT_SIZE = 64
IMG_OF_INTEREST_SIZE = 128
EVAL_CROP_SIZE = IMG_OF_INTEREST_SIZE

DIMENSION_OF_ORIGINAL_IMAGE = 2  # 2 should be Intensity_IW2_VH

# ****** DB ******
# DB creation
VV_CHANNEL = 1
VH_CHANNEL = 0

PERCENTAGE_PICTURES_PER_IMAGE = 1  # 0.1 # calc from img_of_interest_size, rects will be shuffled
PERCENTAGE_EVALUATION = 0.1
PERCENTAGE_TEST = 0.0  # how many % will be used of the train data for testing
PERCENTAGE_TRAIN = 1 - PERCENTAGE_EVALUATION - PERCENTAGE_TEST

IMG_CREATION_STRIDE = 0  # for the grid on which examples are created, n means only every n + 1 are taken
PERCENTAGE_OF_USED_IMAGES = 1

if DEBUG_MODE == MODE_DEBUG:
    PERCENTAGE_OF_USED_IMAGES = 0.01  # speeds up the db creation a lot

col_name_interest = 'interest'
col_name_prediction = 'prediction'
col_name_gc_pre_1 = 'gc_pre_1'
col_name_gc_post_1 = 'gc_post_1'
col_name_gc_pre_2 = 'gc_pre_2'
col_name_gc_post_2 = 'gc_post_2'
col_name_gc_pre_3 = 'gc_pre_3'
col_name_gc_post_3 = 'gc_post_3'

image_to_size = {
    col_name_interest: IMG_OF_INTEREST_SIZE,
    col_name_prediction: IMG_OF_INTEREST_SIZE,
    col_name_gc_pre_1: GLOBAL_CONTEXT_SIZE,
    col_name_gc_post_1: GLOBAL_CONTEXT_SIZE,
    col_name_gc_pre_2: GLOBAL_CONTEXT_SIZE,
    col_name_gc_post_2: GLOBAL_CONTEXT_SIZE,
    col_name_gc_pre_3: GLOBAL_CONTEXT_SIZE,
    col_name_gc_post_3: GLOBAL_CONTEXT_SIZE,
}


# ****** Stuff ******
if DEBUG_MODE != MODE_REAL and DEBUG_MODE != MODE_REAL_INTENSE:
    print("###########################################################")
    for i in range(30):
        print("!!!!!!!!!!!!In Debug Mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("###########################################################")

if USE_GC is False:
    for i in range(30):
        print("!!!!!!!!!!!!NO GC USED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
