from Global_Info import *
import numpy as np
from keras.utils import plot_model
from shutil import rmtree
import os


def save_example(interest, prediction, gc1_pre, gc1_post, gc2_pre, gc2_post, gc3_pre, gc3_post, before, after,
                 before_img, after_img):
    import imageio

    feature = {
        'interest': interest,
        'prediction': prediction,
        'gc1_pre': gc1_pre,
        'gc1_post': gc1_post,
        'gc2_pre': gc2_pre,
        'gc2_post': gc2_post,
        'gc3_pre': gc3_pre,
        'gc3_post': gc3_post,
    }

    func = np.vectorize(lambda x: x * 1000)
    dest = ROOT + 'database/check/'
    for key in feature:
        path = dest + key + '.png'
        # print(key + ' ' + str(feature[key].shape))
        imageio.imwrite(path, func(feature[key]))

    imageio.imwrite(dest + 'befor.png', func(before_img))
    imageio.imwrite(dest + 'after.png', func(after_img))

    print(before)
    print(after)

    print("saved and exited")
    exit(0)


def print_architecture(model):
    model.summary()
    plot_model(model, to_file=MODEL_DIR + 'model.png')
    plot_model(model, show_shapes=True, to_file=MODEL_DIR + 'model_detailed.png')


def get_file_name_date(file_name):
    return int(file_name[3: 3 + len("20180614")])


def get_file_path_date(file_name):
    return get_file_name_date(os.path.split(file_name)[-1])


def print_params():
    params = {'PERCENTAGE_PICTURES_PER_IMAGE': PERCENTAGE_PICTURES_PER_IMAGE,
              'PERCENTAGE_EVALUATION': PERCENTAGE_EVALUATION,
              'PERCENTAGE_TEST_OF_TRAIN': PERCENTAGE_TEST,
              'PERCENTAGE_TRAIN': PERCENTAGE_TRAIN,
              'IMG_CREATION_STRIDE': IMG_CREATION_STRIDE,
              'PERCENTAGE_OF_USED_IMAGES': PERCENTAGE_OF_USED_IMAGES,
              'tifPath': TIF_PATH,
              'trainDatabase': TRAIN_DATABASE,
              'evalDatabase': EVAL_DATABASE,
              'GLOBAL_CONTEXT_SIZE': GLOBAL_CONTEXT_SIZE,
              'IMG_OF_INTEREST_SIZE': IMG_OF_INTEREST_SIZE,
              'DIMENSION_OF_ORIGINAL_IMAGE': DIMENSION_OF_ORIGINAL_IMAGE
              }

    for key in params:
        print(key + ": " + str(params[key]))

    print("###################################")


def prepare_output_folders():
    # mind order
    dirs_to_clean = [PREDICTION_DIR, MODEL_DIR, MODEL_DIR_INBETWEEN_PATH]

    for dir in dirs_to_clean:
        if os.path.exists(dir):
            rmtree(dir)
        os.makedirs(dir)


def print_train_params():
    params = {
        'EPOCHS': EPOCHS,
        'EPOCH_STEP_REGULATOR': EPOCH_STEP_REGULATOR,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'RHO': RHO
    }

    for key in params:
        print(key + ": " + str(params[key]))

    print("###################################")


def print_weights(model):
    conv_of_interest = [1, 2, 3, 4, 5, 13, 14, 15]
    dense_of_interest = [1, 2, 3]

    for layer in model.layers:
        print_layer = None

        for conv in conv_of_interest:
            if layer.name == 'conv2d_' + str(conv):
                print_layer = 1
                break

        for dense in dense_of_interest:
            if layer.name == 'dense_' + str(dense):
                print_layer = 2
            break


        if print_layer is not None:
            print(layer.name)
            weights = layer.get_weights()
            if print_layer == 1:
                array = np.absolute(np.array(weights))
                #print(array)
                print('mean ' + str(array.mean().mean()))
                print('std_dev ' + str(array.std().mean()))
            elif print_layer == 2:
                for a in weights:
                    a = np.absolute(a)
                    print('mean ' + str(a.mean()))
                    print('std_dev ' + str(a.std()))

            print('-------')
