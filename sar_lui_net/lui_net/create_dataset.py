import json
import random
import tensorflow as tf
from multiprocessing import cpu_count
from CreateDatasetWorker import CreateDatasetWorker
from threading import Lock
from time import sleep
from file_pair_creator import *
import evaluation


def create_dataset(database_path, img_pairs):
    print('creating database ' + database_path)

    img_pairs = random.shuffle(img_pairs)
    writer = tf.python_io.TFRecordWriter(database_path)

    writer_lock = Lock()
    workers = []
    start = 0
    worker_count = int(cpu_count() / 2)
    if DEBUG_MODE == MODE_DEBUG:
        worker_count = 1

    files_per_worker = int(len(img_pairs) // worker_count)

    for i in range(worker_count):
        #end = -1 if i + 1 == worker_count else start + files_per_worker
        t = CreateDatasetWorker(img_pairs[start:start + files_per_worker], writer, i, writer_lock)
        t.start()
        workers.append(t)
        start += files_per_worker

    while any(worker.is_alive() for worker in workers):
        sleep(10)

    writer.close()

    example_count = sum(worker.example_count for worker in workers)
    print('100% - done !!!!! Store Numbers of examples in Global Info')
    print(str(example_count) + " examples for DB: " + database_path)


def store_file_list(file_pairs, path):
    serialized = json.dumps(file_pairs)
    with open(path, 'w') as file:
        file.write(serialized)


def split_files_into_train_eval_use_erruptions(files):
    print('For splitting into train and test - I spare all erruptions (from global_info) and use them for evaluation!')

    train_files = list()

    eruptions = evaluation.filter_eruptions_only()

    for pair in files:
        if pair not in eruptions:
            train_files.append(pair)

    return train_files, eruptions


def split_files_into_train_test_eval_randomly(files):
    len_files = len(files)
    files_for_train = int(len_files * PERCENTAGE_TRAIN)
    files_for_test = int(len_files * PERCENTAGE_TEST)
    files_for_eval = int(len_files * PERCENTAGE_EVALUATION)

    train_files = files[0:files_for_train]
    start_idx = files_for_train
    test_files = files[start_idx:start_idx + files_for_test]
    start_idx += files_for_test
    eval_files = files[start_idx: start_idx + files_for_eval]

    return train_files, test_files, eval_files


if __name__ == "__main__":
    print_params()

    file_pairs = create_image_pairs_from_csv_file()

    #train_file_pairs, test_file_pairs, eval_file_pairs = split_files_into_train_test_eval(file_pairs)
    train_file_pairs, eval_file_pairs = split_files_into_train_eval_use_erruptions(file_pairs)
    print('Train Pairs: ' + str(len(train_file_pairs)))
    print('Eval Pairs: ' + str(len(eval_file_pairs)))

    print("Blocked! - So you won't delete your database.")
    exit(0)

    store_file_list(eval_file_pairs, EVAL_FILE_PAIRS)
    store_file_list(train_file_pairs, TRAIN_FILE_PAIRS)

    create_dataset(TRAIN_DATABASE, train_file_pairs)
