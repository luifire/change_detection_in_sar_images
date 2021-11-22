import tensorflow as tf
from Global_Info import *
import warnings
import cv2

#tf.enable_eager_execution()

#  partly copied from https://www.tensorflow.org/tutorials/load_data/tf_records
#  and https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36


def _parse_function(serialized):
    col_names = [col_name_interest, col_name_prediction,
                 col_name_gc_pre_1, col_name_gc_post_1,
                 col_name_gc_pre_2, col_name_gc_post_2]

    features = {name: tf.FixedLenFeature([], tf.string) for name in col_names}

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    prepared_images = []
    for data_name in col_names:
        size = image_to_size[data_name]

        # Get the image as raw bytes.
        image_raw = parsed_example[data_name]

        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.float32)

        image = tf.reshape(image, [2, size, size])
        image = tf.transpose(image, [1, 2, 0])

        prepared_images.append(image)

    return tuple(prepared_images)


def tf_record_size(path):
    c = 0
    for record in tf.python_io.tf_record_iterator(path):
        c += 1
    return c


def _draw_dataset(dataset):
    for image_features in dataset:
        image_raw = image_features[0].numpy()

        cv2.imwrite(image_raw[:, :, 0], MODEL_DIR + "vh.png")
        cv2.imwrite(image_raw[:, :, 1], MODEL_DIR + "vv.png")


def _load_record(path):

    dataset = tf.data.TFRecordDataset(filenames=path)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)

    #_draw_dataset(dataset)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCHS)
    if DEBUG_MODE == MODE_REAL or DEBUG_MODE == MODE_REAL_INTENSE:
        print('everyday I\'m shuffling')
        dataset = dataset.shuffle(buffer_size=BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def load_datasets():
    return _load_record(TRAIN_DATABASE), _load_record(TEST_DATABASE)

