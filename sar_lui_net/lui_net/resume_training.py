from ley_net import *


def resume_training(best_model):

    best_model_path, initial_epoch = best_model

    # Load training and eval data
    train_data, test_data = load_datasets()
    input_img, output_img, gc_pre_1, gc_post_1, gc_pre_2, gc_post_2 = train_data

    ley_net = create_ley_net(input_img, [gc_pre_1, gc_post_1, gc_pre_2, gc_post_2])

    ley_net.load_weights(best_model_path)
    optimizer = keras.optimizers.Adadelta(lr=LEARNING_RATE, rho=RHO, epsilon=1e-06)
    ley_net.compile(loss='mean_squared_error', optimizer=optimizer, target_tensors=output_img)

    steps_per_epoch, steps_per_test = steps_per_epoch_and_validation()
    print_train_params()

    #verbose = 2 if DEBUG_MODE == MODE_REAL_INTENSE else 1
    verbose = 2
    ley_net.fit(initial_epoch=initial_epoch, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                verbose=verbose, callbacks=get_callbacks())


def _find_last_model():
    best_model = ''
    best_model_idx = -1
    for root, dirs, files in os.walk(MODEL_DIR_INBETWEEN_PATH):
        for name in files:
            # saved-model-26-0.657.h5
            if name.endswith('.h5'):
                elements = name.split('-')
                current_idx = int(elements[2])
                if current_idx > best_model_idx:
                    best_model_idx = current_idx
                    best_model = name

    return os.path.join(MODEL_DIR_INBETWEEN_PATH, best_model), best_model_idx


if __name__ == '__main__':
    resume_training(_find_last_model())
