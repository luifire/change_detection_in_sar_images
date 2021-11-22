from __future__ import absolute_import, division, print_function
import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0
import keras
from keras.models import Model
import json

# own
from Global_Info import *
from load_database import load_datasets, tf_record_size
from Helper import *
from layer_block_creator import *

# this is for debugging purposes
#import tensorflow as tf
#tf.enable_eager_execution()
#tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)


def _create_global_context_net(global_context):

    inputs = input_interative(global_context)

    conv_pool_1 = conv_and_pool_iterative(inputs, 32, is_first_layer=False)
    conv_pool_2 = conv_and_pool_iterative(conv_pool_1, 64)
    conv_pool_3 = conv_and_pool_iterative(conv_pool_2, 128)

    if GLOBAL_CONTEXT_SIZE == 64:
        conv_pool_3 = conv_and_pool_iterative(conv_pool_3, 128)  # size / 16 | 4^2 x 128

    last_conv_layer = conv_and_pool_iterative(conv_pool_3, 128)
    # just to make it smaller, could go when we have residual layers
    max_before_dense = max_pool_iterative(last_conv_layer)

    flat_size = get_tensor_size(max_before_dense[0])
    flat = flat_iterative(max_before_dense)
    densors_1 = dense_iterative(flat, flat_size)
    densors_2 = dense_iterative(densors_1, flat_size)
    densors_3 = dense_iterative(densors_2, flat_size // 2)

    fused_gc = fuse_pre_post_iterative(densors_3)
    pooled_output = max_pool_all_gc(fused_gc)

    return inputs, pooled_output


def _fuse_global_context_and_autoencoder(bottleneck, global_context):

    bottle_img_dim = bottleneck.shape.as_list()[1]
    bottle_channel_dim = bottleneck.shape.as_list()[3]
    gc_dim = global_context.shape.as_list()[1]

    # turn [a] into [[a,a], [a,a]] for height and width of bottleneck
    gc_2_d = RepeatVector(bottle_img_dim * bottle_img_dim)(global_context)
    gc_3_d = Reshape((bottle_img_dim, bottle_img_dim, gc_dim))(gc_2_d)

    # concate layers
    concatenated_layers = concatenate([bottleneck, gc_3_d])

    # "fully connected" layer over each cell
    dense_1 = Conv2D(bottle_channel_dim + gc_dim, (1, 1), activation='relu', padding='same')(concatenated_layers)
    dense_2 = Conv2D(bottle_channel_dim + gc_dim, (1, 1), activation='relu', padding='same')(dense_1)
    dense_3 = Conv2D(bottle_channel_dim + gc_dim, (1, 1), activation='relu', padding='same')(dense_2)
    dense_3 = Conv2D(bottle_channel_dim, (1, 1), activation='relu', padding='same')(dense_3)

    return dense_3


def _create_autoencoder(input_layer, global_context_layer):

    if type(input_layer) == tuple:
        input_layer = Input(shape=input_layer, name='input_intereset')
    else:
        input_layer = Input(tensor=input_layer, name='input_intereset')

    ### TEST ####
    #return input_layer, Conv2D(2, (1, 1), activation='linear', padding='same')(input_layer)
    ### Ende TEST ####

    #  Encoder  #
    CONV_START_VAL = 2
    CHANNEL_INCREMENT_STEP = 3.9

    # comments for img_of_intereset_size == 64
    conv_pool_1 = double_resblock_and_pool(input_layer, CONV_START_VAL, is_first_layer=True)  # out: 32 x 32 | in / 2
    conv_pool_2 = double_resblock_and_pool(conv_pool_1, CONV_START_VAL * CHANNEL_INCREMENT_STEP)  # out 16 x 64 | in / 4
    conv_pool_3 = double_resblock_and_pool(conv_pool_2, CONV_START_VAL * CHANNEL_INCREMENT_STEP ** 2)  # out 8 x 128 | in / 8

    #  Fuse Global Context and AutoEncoder  #
    fused_global_context = _fuse_global_context_and_autoencoder(conv_pool_3, global_context_layer)

    #  Decoder  #
    deconv_1 = double_upsample_res_conv(fused_global_context, CONV_START_VAL * CHANNEL_INCREMENT_STEP ** 2)  # min*2 x 64 16
    deconv_2 = double_upsample_res_conv(deconv_1, CONV_START_VAL * CHANNEL_INCREMENT_STEP)  # min*4 x 32 32
    deconv_3 = double_upsample_res_conv(deconv_2, CONV_START_VAL)  # min*8 x 32 64

    # activation linear means a(x) = x
    deconv_4 = conv_3x3(deconv_3, 2, activation='linear', name='output')

    output = deconv_4
    return input_layer, output


def create_ley_net(input_img, global_context):
    gc_inputs, gc_output = _create_global_context_net(global_context)
    auto_input, auto_output = _create_autoencoder(input_img, gc_output)
    ley_net = Model(inputs=[auto_input] + gc_inputs, output=auto_output)

    print('problem with architecture printing')
    #print_architecture(ley_net)
    return ley_net


def get_callbacks():
    # save model frequently
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_DIR_INBETWEEN_FILE, monitor='loss', verbose=0,
                                                 save_best_only=True, save_weights_only=False, mode='min', period=1)
    return [keras.callbacks.History(), checkpoint]


def steps_per_epoch_and_validation():
    #train_size = tf_record_size(TRAIN_DATABASE)
    #test_size = tf_record_size(TEST_DATABASE)
    train_size = TRAIN_EXAMPLE_COUNT
    test_size = TEST_EXAMPLE_COUNT

    steps_per_train = train_size // (BATCH_SIZE * EPOCH_STEP_REGULATOR)
    steps_per_test = test_size // (BATCH_SIZE * EPOCH_STEP_REGULATOR)

    print("Steps per Epoch: " + str(steps_per_train))
    print("Steps per Test: " + str(steps_per_test))

    return steps_per_train, steps_per_test


def init_NN():
    # Load training and eval data
    train_data, test_data = load_datasets()
    input_img, output_img, gc_pre_1, gc_post_1, gc_pre_2, gc_post_2 = train_data

    ley_net = create_ley_net(input_img, [gc_pre_1, gc_post_1, gc_pre_2, gc_post_2])

    # epsilon = fuzzy gradient
    optimizer = keras.optimizers.Adadelta(lr=LEARNING_RATE, rho=RHO, epsilon=1e-06)
    # this one sometimes kicked values to inf
    #optimizer = keras.optimizers.Adagrad(lr=LEARNING_RATE)
    ley_net.compile(loss='mean_squared_error', optimizer=optimizer, target_tensors=output_img)

    steps_per_epoch, steps_per_test = steps_per_epoch_and_validation()
    print_train_params()

    # for testing read bottom comment
    ley_net_train = ley_net.fit(epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=2, callbacks=get_callbacks())

    ley_net.save(MODEL_DIR_SAVE)
    with open(TRAINING_HISTORY_DIR, 'w') as f:
        json.dump(ley_net_train.history, f)

    # multi inputs with tfrecord causes a known keras error
    # https://github.com/tensorflow/tensorflow/issues/20698
    # validation_data=([test_input_layer, test_gc_pre_layer, test_gc_post_layer]
    # validation_data = (test_input_img, test_output_img),
    # validation_steps = steps_per_test,


if __name__ == '__main__':
    print("blocked!")
    exit(1)
    prepare_output_folders()
    init_NN()
