from Global_Info import *
from os import path
import csv
import random

from Helper import *


def create_image_pairs_from_csv_file(csv_file_path=CSV_CLEANED_FILES, root_path=TIF_PATH):
    image_groups = dict()

    volcano = 0
    orbit = 1
    file_name = 2
    group = 3

    with open(csv_file_path) as csvfile:
        ordered_files = csv.reader(csvfile, delimiter=';')
        for row in ordered_files:
            # they are bad
            if int(row[group]) == -1:
                continue

            group_name = row[volcano] + row[orbit] + '-' + row[group]
            if group_name not in image_groups:
                image_groups[group_name] = list()

            image_groups[group_name].append(row)

    def row_to_file_path(row):
        return os.path.join(root_path, row[volcano], row[orbit], row[file_name]) + '.tif'

    for key in image_groups:
        image_groups[key].sort(key=lambda item: (get_file_name_date(item[file_name])))

    file_pairs = list()
    for key in image_groups:
        image_group = image_groups[key]
        for idx, images in enumerate(image_group):
            # no more consecutive image
            if idx + 1 == len(image_group):
                break

            date_1 = get_file_name_date(image_group[idx][file_name])
            date_2 = get_file_name_date(image_group[idx + 1][file_name])

            # as many pictures would be lost otherwise, we actually at maximum compare images 30 days apart
            if date_2 - date_1 <= 30:
                file_pairs.append((row_to_file_path(image_group[idx]), row_to_file_path(image_group[idx + 1])))

    print('Found ' + str(len(file_pairs)) + ' file pairs.')
    return file_pairs


def create_image_pairs_in_folder(folder, files):

    # so dates are ordered
    # S1_20180614T001338_136_int
    files.sort(key=lambda item: (int(item[3:3 + len("20180614")])))

    def add_path(file): return os.path.join(folder, file)
    lst = []
    for i in range(len(files)):
        if i < len(files) - 1:
            lst.append((add_path(files[i]), add_path(files[i+1])))
    return lst


def create_all_file_pairs(path=TIF_PATH):
    pair_list = []

    """ walk over all tiffs in the given directory"""
    for root, dirs, files in os.walk(path):
        tif_files = []
        for name in files:
            if name.endswith('.tif'):
                tif_files.append(name)

        if len(tif_files) > 0:
            pair_list += create_image_pairs_in_folder(root, tif_files)

    print("Total file pairs found: " + str(len(pair_list)))
    # so they will be party in eval and train
    random.shuffle(pair_list)
    pair_list = pair_list[0:int(len(pair_list) * PERCENTAGE_OF_USED_IMAGES)]
    print("Total file pairs used: " + str(len(pair_list)))

    return pair_list


def _split_file_pairs_into_orbit(file_pairs):
    orbits = dict()

    for pair in file_pairs:
        normed_path = path.normpath(pair[0])
        splitted_path = normed_path.split(path.sep)

        orbit = splitted_path[-2]

        if orbit not in orbits:
            orbits[orbit] = list()

        orbits[orbit].append(pair)

    return orbits


def load_file_pair_of_period(volcano, start_date, end_date, path=TIF_PATH):
    img_pair = create_image_pairs_from_csv_file(root_path=path)

    def filter_volcanoes(volcano_pair):
        if volcano.lower() in volcano_pair[0].lower() and \
            start_date <= get_file_path_date(volcano_pair[0]) <= end_date and \
                start_date <= get_file_path_date(volcano_pair[1]) <= end_date:
            return True

        return False

    # only take volcano in wanted period
    filtered_volcanoes = list(filter(filter_volcanoes, img_pair))

    filtered_volcanoes = _split_file_pairs_into_orbit(filtered_volcanoes)

    return filtered_volcanoes
