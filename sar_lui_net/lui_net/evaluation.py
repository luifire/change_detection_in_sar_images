import warnings

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0

import json
from keras.models import Model, load_model
from keras import backend as K
import cv2
import ley_net

# own
from Helper import *
from Global_Info import *
from CreateDatasetWorker import *
from file_pair_creator import create_all_file_pairs, create_image_pairs_from_csv_file


TOP = 0
LEFT = 1
RIGHT = 2
BOTTOM = 3

abnormal_images = []


def load_eval_file_pairs_from_file():
    with open(EVAL_FILE_PAIRS) as file_pairs_json:
        json_str = file_pairs_json.read()
        file_pairs = json.loads(json_str)

    return file_pairs


def create_color_diff_image(predicted_images, squared_img, rects, polarisation_channel):
    width = squared_img.shape[1]
    height = squared_img.shape[0]

    diff_img = np.zeros((height, width, 3))
    # future - predicted
    diff = predicted_images[1][:, :, polarisation_channel] - predicted_images[0][:, :, polarisation_channel]

    scaling = 255
    for x in range(width):
        for y in range(height):
            val = diff[y, x]
            if val > 0:
                diff_img[y, x, 0] = squared_img[y, x] * scaling
            else:
                diff_img[y, x, 2] = squared_img[y, x] * scaling

    diff_img = draw_rects_in_img(diff_img, rects)
    return diff_img


def create_color_diff_image_threshold(path, squared_diff_img, rects):
    width = squared_diff_img.shape[1]
    height = squared_diff_img.shape[0]

    diff_img = np.zeros((height, width, 3))

    sum = 0
    for x in range(width):
        for y in range(height):
            val = squared_diff_img[y, x]
            pixel_val = 0
            if val[0] > THRESHOLD_PER_PIXEL:
                pixel_val = val[0]

            if val[1] > THRESHOLD_PER_PIXEL:
                pixel_val += val[1]

            if pixel_val > 0:
                sum += pixel_val
                diff_img[y, x, 0] = pixel_val * 255

    diff_img = draw_rects_in_img(diff_img, rects)
    cv2.imwrite(path, diff_img)

    return diff_img, sum / (height * width * 2)


def draw_rects_in_img(img, rects):
    # make it a bit smaller so that it will definitely not influence the brightness of the picture
    draw_value = np.max(img) * 0.8

    # so we can draw colored rectangles
    img = make_image_3_channeled(img)

    for rect in rects:
        top = rect[TOP]
        left = rect[LEFT]
        bottom = rect[BOTTOM]
        right = rect[RIGHT]

        for boarder in range(3):
            img[top:bottom, left + boarder, 1] = 0
            img[top:bottom, right - boarder, 1] = 0
            img[top + boarder, left:right, 1] = 0
            img[bottom - boarder, left:right, 1] = 0

            for channel in [0, 2]:
                img[top:bottom, left + boarder, channel] = draw_value
                img[top:bottom, right - boarder, channel] = draw_value
                img[top + boarder, left:right, channel] = draw_value
                img[bottom - boarder, left:right, channel] = draw_value

    return img


def _to_normal_date(date):
    date = str(date)
    return date[0:4] + '.' + date[4:6] + '.' + date[6:8]


def save_abnormal_change(path_1, img_1, path_2, img_2, votes, total_mse, salient_rects, predicted_images):
    print('Threshold crossed abnormal image votes: ' + str(votes) + " total_mse: " + str(total_mse))
    img_1, img_2 = make_images_same_size(img_1, img_2)

    dir_1, file_1 = os.path.split(path_1)
    dir_2, file_2 = os.path.split(path_2)
    file_date = get_file_name_date(file_2)

    dir_1 = os.path.normpath(dir_1)
    eval_dir = PREDICTION_DIR + dir_1.split(os.path.sep)[-2] + '/' + _to_normal_date(file_date) + ' votes-' + str(votes) \
               + '  mse-' + str(total_mse)[0:7] + '  ' + file_1[:-3] + " - " + file_2[:-3]
    if os.path.exists(eval_dir) is False:
        os.makedirs(eval_dir)

    eval_dir += '/'

    diff_path = eval_dir
    abnormal_images.append((path_1, path_2, diff_path, eval_dir, salient_rects))

    squared_diff_img = (predicted_images[1] - predicted_images[0]) ** 2
    for channel, polarisation_name in [(VV_CHANNEL, 'vv_'), (VH_CHANNEL, 'vh_')]:
        # diff images
        diff_img = create_color_diff_image(predicted_images, squared_diff_img[:, :, channel], salient_rects, channel)
        cv2.imwrite(eval_dir + polarisation_name + 'diff_.png', diff_img)

        # predicted images
        tmp = draw_rects_in_img(predicted_images[0][:, :, channel] * 255, salient_rects)
        cv2.imwrite(eval_dir + polarisation_name + 'predicted_image.png', tmp)
        tmp = draw_rects_in_img(predicted_images[1][:, :, channel] * 255, salient_rects)
        cv2.imwrite(eval_dir + polarisation_name + 'predicted_future.png', tmp)

    diff_img_thresh, threshold_crosses = create_color_diff_image_threshold(eval_dir + 'diff_img_thresh.png',
                                                                           squared_diff_img, salient_rects)

    # draw markers
    picture_list = {
        'vh - 1 intereset': (img_1, VH_CHANNEL),
        'vv - 1 intereset': (img_1, VV_CHANNEL),
        'vh - 2 future': (img_2, VH_CHANNEL),
        'vv - 2 future': (img_2, VV_CHANNEL),
    }
    new_width = predicted_images[0].shape[1]
    new_height = predicted_images[0].shape[0]
    for polarisation_name in picture_list:
        img, channel = picture_list[polarisation_name]
        img = img[channel, 0:new_height, 0:new_width]

        img = draw_rects_in_img(img, salient_rects)

        tiff.imwrite(eval_dir + polarisation_name + '.tif', img)

    return threshold_crosses
    # save original images
    """org_img_1 = read_image(path_1)
    org_img_2 = read_image(path_2)
    tiff.imwrite(eval_dir + file_1, org_img_1)
    tiff.imwrite(eval_dir + file_2, org_img_2)
    
    #copyfile(path_1, eval_dir + file_1)
    #copyfile(path_2, eval_dir + file_2)
    """


def reshape_image(img):
    img = np.moveaxis(img, 0, 2)
    img = np.expand_dims(img, axis=0)
    return img


def write_crop_into_image(img, crop, rect):
    tmp = np.squeeze(crop[:, :, :, :])
    img[rect.top:rect.bottom, rect.left:rect.right, :] = tmp


def sliding_window_evaluation(diff_img, x_offset, y_offset):
    mse_sum = 0
    highest_mse = -1
    abnormal_rects = []
    crops_in_img = diff_img.shape[1] // EVAL_CROP_SIZE

    for i in range(crops_in_img):
        start = i * EVAL_CROP_SIZE
        end = start + EVAL_CROP_SIZE
        small_diff_img = diff_img[:, start:end, start:end, :]
        mse_small_diff_img = small_diff_img.mean()
        mse_sum += mse_small_diff_img
        if mse_small_diff_img >= THRESHOLD_MSE_PER_CROP:
            store = dict()
            store[TOP] = y_offset + start
            store[BOTTOM] = y_offset + end

            store[LEFT] = x_offset + start
            store[RIGHT] = x_offset + end

            abnormal_rects.append(store)

        if mse_small_diff_img > highest_mse:
            highest_mse = mse_small_diff_img

    return highest_mse, abnormal_rects, mse_sum / crops_in_img


def evaluate(ley_net_model, file_pair, save_abnormal_files):

    def slice_and_reshape(img, rect_slice, size):
        img = slice_image(img, rect_slice, size)
        img = reshape_image(img)
        return img

    before_img = read_sar_image(file_pair[0])
    after_img = read_sar_image(file_pair[1])

    if after_img is None or before_img is None or \
            check_equal_dimensions(before_img, after_img, file_pair[0], file_pair[1]) is False:
        return

    before_img = NormalizeData.normalize_image(before_img)
    after_img = NormalizeData.normalize_image(after_img)

    before_img, after_img = make_images_same_size(before_img, after_img)

    # as we will only draw in the grid
    shape = ((before_img.shape[1] // IMG_OF_INTEREST_SIZE) * IMG_OF_INTEREST_SIZE + 3,
             (before_img.shape[2] // IMG_OF_INTEREST_SIZE) * IMG_OF_INTEREST_SIZE + 3,
             2)
    predicted_image = np.zeros(shape)
    predicted_future = np.zeros(shape)

    highest_threshold_cross = 0
    total_mse = 0
    salient_rects = []
    rects = create_rectangles(before_img, rectangle_count=AMOUNT_OF_GLOBAL_CONTEXTS, for_training=False)
    for rect in rects:
        interest = slice_and_reshape(before_img, rect[0], IMG_OF_INTEREST_SIZE)
        future = slice_and_reshape(after_img, rect[0], IMG_OF_INTEREST_SIZE)

        global_context = list()
        global_context_future = list()
        # 1 because 0 is the image of interest
        for i in range(1, len(rect)):
            gc_pre = slice_and_reshape(before_img, rect[i], GLOBAL_CONTEXT_SIZE)
            gc_post = slice_and_reshape(after_img, rect[i], GLOBAL_CONTEXT_SIZE)

            if USE_GC:
                global_context += [gc_pre, gc_post]
            else:
                global_context += [gc_pre, gc_pre]
            global_context_future += [gc_post, gc_post]

        prediction = ley_net_model.predict([interest] + global_context)

        write_crop_into_image(predicted_image, prediction, rect[0])
        # predit future picture to get closer results
        if PREDICT_FUTURE:
            future = ley_net_model.predict([future] + global_context_future)
            write_crop_into_image(predicted_future, future, rect[0])

        diff_img = (prediction - future) ** 2
        highest_mse, abnormal_rects, mse_sum = sliding_window_evaluation(diff_img, x_offset=rect[0].left, y_offset=rect[0].top)

        total_mse += mse_sum
        salient_rects = salient_rects + abnormal_rects

        if highest_mse > highest_threshold_cross:
            highest_threshold_cross = highest_mse

    votes = len(salient_rects)
    print("votes: " + str(votes) + " total: " + str(total_mse / len(rects))[0:5] + " " +
          file_pair[0] + "  -  " + file_pair[1])

    threshold_crosses = 0
    if save_abnormal_files:
        if 1 <= votes < THRESHOLD_VOTES_PER_IMAGE:
        #if 1 <= votes:
        #if True:
            threshold_crosses = save_abnormal_change(file_pair[0], before_img, file_pair[1], after_img, votes,
                                                     total_mse / len(rects), salient_rects,
                                                     (predicted_image, predicted_future))

    # total_mse is the sum of the mses per interest region
    return total_mse / len(rects), threshold_crosses


def store_abnormal_images():
    serialized = json.dumps(abnormal_images)
    with open(PREDICTION_DIR + 'abnormal_images.json', 'w') as file:
        file.write(serialized)


def filter_eruptions_only(tif_path=TIF_PATH):
    #warnings.warn('Using all file pairs!!!!!!!')
    #file_list = create_all_file_pairs()
    file_list = create_image_pairs_from_csv_file(root_path=tif_path)
    #file_list = load_file_pairs()

    def file_within_range(file, key):
        if key.lower() in file.lower():
            dir, file_name = os.path.split(file)
            date = int(file_name[3:3 + len("20180614")])

            start, end = VOLCANOES_OF_INTEREST[key]
            if start <= date <= end:
                return True

        return False

    new_list = list()
    for pair in file_list:
        for key in VOLCANOES_OF_INTEREST:
            if file_within_range(pair[0], key) and file_within_range(pair[1], key):
                new_list.append(pair)

    return new_list


def load_model_increase_gc():
    K.set_learning_phase(0)
    interest_shape = (IMG_OF_INTEREST_SIZE, IMG_OF_INTEREST_SIZE, 2)
    gc_shape = (GLOBAL_CONTEXT_SIZE, GLOBAL_CONTEXT_SIZE, 2)
    gc_inputs = [gc_shape for i in range(AMOUNT_OF_GLOBAL_CONTEXTS * 2)]

    trained_net = load_model(MODEL_DIR_SAVE)
    ley_net_model = ley_net.create_ley_net(interest_shape, gc_inputs)
    ley_net_model.set_weights(trained_net.get_weights())

    #print_weights(ley_net_model)
    return ley_net_model


def evaluation():
    warnings.warn('Using alternative tif path')
    #file_pairs = load_file_pairs()
    #file_pairs = create_all_file_pairs(ALTERNATIVE_TIF_PATH)
    file_pairs = filter_eruptions_only(ALTERNATIVE_TIF_PATH)

    def cmp(x):
        return x[0]

    file_pairs = sorted(file_pairs, key=cmp)

    print('Looking at ' + str(len(file_pairs)) + ' file pairs!')
    # clean dir
    if os.path.exists(PREDICTION_DIR):
        rmtree(PREDICTION_DIR)
    os.makedirs(PREDICTION_DIR)

    ley_net_model = load_model_increase_gc()
    for i, file_pair in enumerate(file_pairs):
        print(str(i / len(file_pairs) * 100)[0:4] + "%")
        evaluate(ley_net_model, file_pair, save_abnormal_files=True)
        # for quicker evaluation, also we might not look at all pictures
        if i % 5 == 0:
            store_abnormal_images()

    store_abnormal_images()


if __name__ == '__main__':
    # so we won't have any batch norm
    evaluation()

