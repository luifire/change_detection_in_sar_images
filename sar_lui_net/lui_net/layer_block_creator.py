from keras.layers import *
import resnet


def _apply_layer_iterative(layer, inputs):
    return [layer(inp) for inp in inputs]


def upsample_res_conv(input_tensor, channel_size, activation='relu'):
    channel_size = int(channel_size)
    upsample = UpSampling2D((2, 2))(input_tensor)  # (size * 2)^2 add zeros
    #deconv = Conv2D(channel_size, (3, 3), activation=activation, padding='same')(upsample)  # (size*2)^2 * channel_size

    res_block = resnet._residual_block(resnet.basic_block, filters=channel_size, blocks=2,
                                       stage=2, is_first_layer=False)(upsample)
    return res_block

def upsample_res_conv(input_tensor, channel_size, activation='relu'):
    channel_size = int(channel_size)
    upsample = UpSampling2D((2, 2))(input_tensor)  # (size * 2)^2 add zeros
    #deconv = Conv2D(channel_size, (3, 3), activation=activation, padding='same')(upsample)  # (size*2)^2 * channel_size
    res_block = resblock(upsample, channel_size, block_count=2, is_first_layer=False)

    return res_block


def double_upsample_res_conv(input_tensor, channel_size):
    channel_size = int(channel_size)
    upsample = UpSampling2D((2, 2))(input_tensor)  # (size * 2)^2 add zeros
    #deconv = Conv2D(channel_size, (3, 3), activation=activation, padding='same')(upsample)  # (size*2)^2 * channel_size
    res_block = resblock(upsample, channel_size, block_count=2, is_first_layer=False)
    res_block = resblock(res_block, channel_size, block_count=2, is_first_layer=False)
    res_block = resblock(res_block, channel_size, block_count=2, is_first_layer=False)

    return res_block


def conv_3x3(input_tensor, channel_size, activation='relu', name=None):
    return Conv2D(channel_size, (3, 3), activation=activation, padding='same', name=name)(input_tensor)  # input size^2 * channel_size


def pool_2x2(input_tensor):
    return MaxPooling2D(pool_size=(2, 2))(input_tensor)  # out: input size/2 x channel_size


def double_resblock_and_pool(input, channel_size, block_count=2, is_first_layer=False):
    res_block = resblock(input, channel_size, block_count, is_first_layer)
    res_block = resblock(res_block, channel_size, block_count, False)
    res_block = resblock(res_block, channel_size, block_count, False)
    pool = pool_2x2(res_block)

    return pool


def resblock_and_pool(input, channel_size, block_count=2, is_first_layer=False):
    res_block = resblock(input, channel_size, block_count, is_first_layer)
    pool = pool_2x2(res_block)

    return pool

def resblock(input, channel_size, block_count=2, is_first_layer=False):
    channel_size = int(channel_size)
    res_block = resnet._residual_block(resnet.basic_block, filters=channel_size, blocks=block_count,
                                       stage=2, is_first_layer=is_first_layer)(input)
    return res_block


def input_interative(inputs):
    if type(inputs[0]) == tuple:
        return [Input(shape=input) for input in inputs]
    else:
        return [Input(tensor=input) for input in inputs]


def conv_and_pool_iterative(inputs, channel_size, is_first_layer=False):
    # unfortunately this doesn't work for shared weights
    # if someone wants, follow the path and reimplement it with shared weights
    #res_block_layer = resnet._residual_block(resnet.basic_block, filters=channel_size, blocks=1,
     #                                  stage=_stage, is_first_layer=is_first_layer)
    conv_layer = Conv2D(channel_size, (3, 3), activation='relu', padding='same')
    conv_tensors = _apply_layer_iterative(conv_layer, inputs)

    return max_pool_iterative(conv_tensors)


def max_pool_iterative(inputs):
    max_layer = MaxPooling2D(pool_size=(2, 2))
    return _apply_layer_iterative(max_layer, inputs)


def average_pool_all_gc(inputs):
    gc_count = len(inputs)
    concat_all_gc = concatenate(inputs)
    reshaped = Reshape((inputs[0].shape.as_list()[1], gc_count))(concat_all_gc)
    averaged = AveragePooling1D(pool_size=gc_count, data_format='channels_first')(reshaped)
    float_output = Reshape(target_shape=(averaged.shape.as_list()[1], ))(averaged)

    return float_output


def max_pool_all_gc(inputs):
    gc_count = len(inputs)
    concat_all_gc = concatenate(inputs)
    reshaped = Reshape((inputs[0].shape.as_list()[1], gc_count))(concat_all_gc)

    pooled = GlobalMaxPooling1D(data_format='channels_first')(reshaped)  # out: input size/2 x channel_size

    return pooled


def flat_iterative(inputs):
    flat_layer = Flatten()

    outputs = _apply_layer_iterative(flat_layer, inputs)
    return outputs


def get_tensor_size(tensor):
    input_shape = tensor.shape.as_list()
    product = 1
    for shape in input_shape[1:len(input_shape)]:
        product = product * shape
    return product


def dense_iterative(inputs, channel_size):
    dense_layer = Dense(channel_size, activation='relu')
    return _apply_layer_iterative(dense_layer, inputs)


def fuse_pre_post_iterative(inputs):
    paired_input = list()
    for idx in range(len(inputs) // 2):
        paired_input.append((inputs[idx * 2], inputs[idx * 2 + 1]))

    subtracted_layer = Subtract()
    subtracted_tensors = list()
    for tensor_pre, tensor_post in paired_input:
        subtracted_tensor = subtracted_layer([tensor_pre, tensor_post])
        subtracted_tensors.append(subtracted_tensor)

    return subtracted_tensors


def flat_and_dense_merge_iterative(inputs):
    paired_input = list()

    for idx in range(len(inputs) // 2):
        paired_input.append((inputs[idx * 2], inputs[idx * 2 + 1]))

    input_shape = inputs[0].shape.as_list()
    flat_size = input_shape[1] * input_shape[2] * input_shape[3] * 2  # * 2 as we concat two inputs

    flat_layer = Flatten()
    dense_1 = Dense(flat_size, activation='relu')
    dense_2 = Dense(flat_size, activation='relu')

    outputs = list()
    for tensor_pre, tensor_post in paired_input:
        tensor_pre = flat_layer(tensor_pre)
        tensor_post = flat_layer(tensor_post)

        concatenated = concatenate([tensor_pre, tensor_post])

        densor_1 = dense_1(concatenated)
        densor_2 = dense_2(densor_1)

        outputs.append(densor_2)

    return outputs
