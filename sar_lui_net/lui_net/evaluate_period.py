import warnings
import csv
import os
from datetime import date, timedelta

# own
from Helper import *
from Global_Info import *
from CreateDatasetWorker import *
from file_pair_creator import load_file_pair_of_period
from evaluation import load_model_increase_gc, evaluate


def _volcano_date_to_date(volcano):
    # 20180618
    start_month = 4
    start_day = start_month + 2
    volcano = str(volcano)

    return date(int(volcano[0:4]), int(volcano[start_month:start_month + 2]), int(volcano[start_day:start_day + 2]))


# this function adds a new line
# it also appends missing dates, in case there are some missing
# every 6 days we want to see a new volcano
def _append_to_csv(csv_list, mse, threshold_crosses, file_pair):
    six_days = timedelta(days=6)
    date_start = get_file_path_date(file_pair[0])
    date_end = get_file_path_date(file_pair[1])

    date_start = _volcano_date_to_date(date_start)
    date_end = _volcano_date_to_date(date_end)

    # append missing dates
    # this is a bit hacky :/
    if len(csv_list) > 1 and len(csv_list[-1]) > 1:
        # restore last end date
        last_end_date = csv_list[-1][1]
        if type(last_end_date) == date:
            while last_end_date + six_days < date_start:
                last_end_date = last_end_date + six_days
                #csv_list.append([last_end_date, last_end_date, 0, 0])

    csv_list.append([date_start, date_end, mse, threshold_crosses])


def evaluate_period(volcano, start, end, ley_net_model):
    orbit_file_pairs = load_file_pair_of_period(volcano, start, end, ALTERNATIVE_TIF_PATH)
    csv_path = BRANCH_DIR + volcano + ' ' + str(start) + '-' + str(end) + '.csv'

    output = []

    for orb_idx, orbit in enumerate(orbit_file_pairs):
        #if orbit != '144':
         #   print('only use 144 orbit for now')
         #   continue

        output.append([orbit])
        output.append(['start', 'end', 'mse', 'mse + thres'])

        file_pairs = orbit_file_pairs[orbit]
        for pair_idx, pair in enumerate(file_pairs):
            print('orbit ' + str(orb_idx + 1) + '/' + str(len(orbit_file_pairs)) +
                  ' - files ' + str(pair_idx + 1) + '/' + str(len(file_pairs)))
            mse, threshold_crosses = evaluate(ley_net_model, pair, save_abnormal_files=True)
            _append_to_csv(output, mse, threshold_crosses, pair)

        output.append([])
        output.append([])

        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';')
            for row in output:
                csv_writer.writerow(list(map(str, row)))


if __name__ == '__main__':
    # clean dir
    if os.path.exists(PREDICTION_DIR):
        rmtree(PREDICTION_DIR)
        pass

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)

    ley_net_model = load_model_increase_gc()

    #evaluate_period('ambrym', 20181213, 20181219, ley_net_model)
    #evaluate_period('ambrym', 20180920, 20190416, ley_net_model)
    #evaluate_period('ambrym', 20190215, 20190227, ley_net_model)
    #evaluate_period('pitonfournaise', 20180305, 20190501, ley_net_model)
    #evaluate_period('pitonfournaise', 20180316, 20180328, ley_net_model)

    #evaluate_period('pitonfournaise', 20180501, 20180630, ley_net_model)
    #evaluate_period('pitonfournaise', 20190201, 20190415, ley_net_model)
    #evaluate_period('pitonfournaise', 20180901, 20181130, ley_net_model)

    #evaluate_period('ambrym', 20181201, 20190201, ley_net_model)
    evaluate_period('ambrym', 20190112, 20190118, ley_net_model)
