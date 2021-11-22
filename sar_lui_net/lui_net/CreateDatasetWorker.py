
from threading import Thread, Lock
import NormalizeData
import random
import tensorflow as tf
from PointsAndRectangles import *
from image_manipulation import *


def read_image_and_normalize(image_path):
    img = read_sar_image(image_path)
    if img is None:
        return None

    normalized = NormalizeData.normalize_image(img)
    return normalized


def slice_image(img, rect, size=-1):
    a = img[:, rect.top:rect.bottom, rect.left:rect.right]

    if size != -1:
        if a.shape[1] != size or a.shape[2] != size or \
                rect.bottom - rect.top != size or rect.right - rect.left != size:
            print(a.shape)
            print(rect)
            warnings.warn("Something is not right with the shapes!")
            exit(2)
    return a


def create_rectangles(image, rectangle_count, percentage_pictures_per_image=1, for_training=True):

    height = image.shape[1]
    width = image.shape[2]

    if height < IMG_OF_INTEREST_SIZE or width < IMG_OF_INTEREST_SIZE:
        warnings.warn('can not use image, too small')
        return

    def get_random_top_bottom_boarder_rect():
        x_pos = random.randint(0, width - GLOBAL_CONTEXT_SIZE)
        y_pos = 0 if random.randint(0, 1) == 0 else height - GLOBAL_CONTEXT_SIZE

        return Rect(x_pos, y_pos, GLOBAL_CONTEXT_SIZE, GLOBAL_CONTEXT_SIZE)

    def get_random_left_right_boarder_rect():
        y_pos = random.randint(0, height - GLOBAL_CONTEXT_SIZE)
        x_pos = 0 if random.randint(0, 1) == 0 else width - GLOBAL_CONTEXT_SIZE

        return Rect(x_pos, y_pos, GLOBAL_CONTEXT_SIZE, GLOBAL_CONTEXT_SIZE)

    def get_random_rect():
        y_pos = random.randint(0, height - GLOBAL_CONTEXT_SIZE)
        x_pos = random.randint(0, width - GLOBAL_CONTEXT_SIZE)

        return Rect(x_pos, y_pos, GLOBAL_CONTEXT_SIZE, GLOBAL_CONTEXT_SIZE)

    def get_not_overlapping_rect(rects, rect_generator):
        random_rect = rect_generator()

        for rect in rects:
            if random_rect.overlaps(rect):
                return get_not_overlapping_rect(rects, rect_generator)

        return random_rect

    entire_rect = Rect(0, 0, width=width, height=height)

    rectangle_generators = [get_random_rect, get_random_top_bottom_boarder_rect,
                            get_random_left_right_boarder_rect, get_random_left_right_boarder_rect]
    x = 0
    y = 0
    list_of_rects = []
    # while we can still move lines downward
    while y < height:
        rect_of_img = Rect(x=x * IMG_OF_INTEREST_SIZE, y=y * IMG_OF_INTEREST_SIZE,
                           width=IMG_OF_INTEREST_SIZE, height=IMG_OF_INTEREST_SIZE)

        if entire_rect.contains_rect(rect_of_img):
            rects = list()

            # interest
            rects.append(rect_of_img)

            # global context
            for i in range(rectangle_count):
                rectangle_generator = rectangle_generators[random.randint(0, len(rectangle_generators) - 1)]
                rects.append(get_not_overlapping_rect(rects, rectangle_generator))

            list_of_rects.append(rects)

            # go to next column
            x += 1 + IMG_CREATION_STRIDE
        else:
            # go to next row
            y += 1 + IMG_CREATION_STRIDE
            x = 0

    # add as many random rects as came from the grid itself
    if for_training:
        for i in range(len(list_of_rects)):
            rects = list()

            y = random.randint(0, height - IMG_OF_INTEREST_SIZE)
            x = random.randint(0, width - IMG_OF_INTEREST_SIZE)
            # don't add the same image again
            if x % IMG_OF_INTEREST_SIZE == 0 or y % IMG_OF_INTEREST_SIZE == 0:
                continue

            main_rect = Rect(x=x, y=y, width=IMG_OF_INTEREST_SIZE, height=IMG_OF_INTEREST_SIZE)
            rects.append(main_rect)
            # global context
            for i in range(rectangle_count):
                rectangle_generator = rectangle_generators[random.randint(0, len(rectangle_generators) - 1)]
                rects.append(get_not_overlapping_rect(rects, rectangle_generator))

            list_of_rects.append(rects)

    if for_training:
        random.shuffle(list_of_rects)

    list_of_rects = list_of_rects[0:int(len(list_of_rects) * percentage_pictures_per_image)]

    return list_of_rects


def _bytes_feature_of_img(img):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))


def check_equal_dimensions(img_1, img_2, path_1, path_2):
    if abs(img_1.shape[1] - img_2.shape[1]) > 10 \
            or abs(img_1.shape[2] - img_2.shape[2]) > 10:
        warnings.warn('following images have unequal dimensions (wont use them)' +
                      str(img_1.shape) + str(img_2.shape) + '  -  ' + path_1 + '  -  ' + path_2)
        return False

    return True


class CreateDatasetWorker(Thread):

    def __init__(self, img_pairs, writer, worker_id, mutex):
        self._img_pairs = img_pairs
        self._writer = writer
        self._worker_id = worker_id
        self._mutex = mutex
        self.example_count = 0

        Thread.__init__(self)

    def run(self):
        # global total_set
        for i in range(len(self._img_pairs)):
            print("Worker " + str(self._worker_id) + ": " + str(i / len(self._img_pairs) * 100.0)[0:4] + '%')

            before_path, after_path = self._img_pairs[i]

            before_img = read_sar_image(before_path)
            after_img = read_sar_image(after_path)

            #  some images are corrupted
            if before_img is None or after_img is None or \
                    check_equal_dimensions(before_img, after_img, before_path, after_path) is False:
                continue

            before_img = NormalizeData.normalize_image(before_img)
            after_img = NormalizeData.normalize_image(after_img)

            # pad images with zeros to make them the same size
            before_img, after_img = make_images_same_size(before_img, after_img)

            list_of_rects = create_rectangles(before_img, rectangle_count=2,
                                              percentage_pictures_per_image=PERCENTAGE_PICTURES_PER_IMAGE)
            for rects in list_of_rects:
                self.example_count += 1
                interest = slice_image(after_img, rects[0], IMG_OF_INTEREST_SIZE)
                prediction = slice_image(before_img, rects[0], IMG_OF_INTEREST_SIZE)

                gc1_pre = slice_image(before_img, rects[1], GLOBAL_CONTEXT_SIZE)
                gc1_post = slice_image(after_img, rects[1], GLOBAL_CONTEXT_SIZE)

                gc2_pre = slice_image(before_img, rects[2])
                gc2_post = slice_image(after_img, rects[2])

                # gc3_pre = slice_image(before_img, rects[3])
                # gc3_post = slice_image(after_img, rects[3])

                feature = {
                    col_name_interest: _bytes_feature_of_img(interest),
                    col_name_prediction: _bytes_feature_of_img(prediction),

                    col_name_gc_pre_1: _bytes_feature_of_img(gc1_pre),
                    col_name_gc_post_1: _bytes_feature_of_img(gc1_post),
                    col_name_gc_pre_2: _bytes_feature_of_img(gc2_pre),
                    col_name_gc_post_2: _bytes_feature_of_img(gc2_post),
                    # col_name_gc_pre_3: _bytes_feature_of_img(gc3_pre),
                    # col_name_gc_post_3: _bytes_feature_of_img(gc3_post),
                }

                # save_example(interest, prediction, gc1_pre, gc1_post, gc2_pre, gc2_post, gc3_pre, gc3_post,
                #            before, after, before_img, after_img)

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                self._mutex.acquire()
                # Serialize to string and write on the file
                self._writer.write(example.SerializeToString())
                self._mutex.release()

        print("################### Worker " + str(self._worker_id) + " finished successfully")


def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
