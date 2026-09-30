#!/usr/bin/env python3
from algorithms.ELHACNP_train import execute_pre_tuned_alns_cnp
import matplotlib.pyplot as plt
from scipy.spatial.distance import chebyshev
import scienceplots
import numpy as np
import json
import pickle

plt.style.use(['science', 'ieee'])

class Executor:
    """
    Experiment Repeater
    """
    operator_weights: list

    def __init__(self, n, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.execute_times = n
        self.operator_weights = []

    def run_function_n_times(self):
        """
        run experiment repeat n times
        :return:
        """
        for _ in range(self.execute_times):
            operator_weight = self.func(*self.args, **self.kwargs)
            self.operator_weights.append(operator_weight)

    def analyse_result(self):
        """
        calculate the mean of operator weights
        :return:
        """
        combined_array = np.stack(self.operator_weights)
        mean_operator_weights = np.mean(combined_array, axis=0)
        return mean_operator_weights

    def reset_container(self):
        """
        reset result container
        :return:
        """
        self.operator_weights = []

class JsonModifier:
    """
    Experiment Parameter Modifier
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        load the json
        :return:
        """
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def write_data(self, data):
        """
        update the information in json
        :param data:
        :return:
        """
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)

    def modify_and_save(self, updates_dict):
        """
        call this function to execute
        :param updates_dict:
        :return:
        """
        data = self.load_data()
        for key, new_value in updates_dict.items():
            data[key] = new_value
        self.write_data(data)

class BreakLoop(Exception):pass

def run_and_analyse(executor):
    """
    MetaExecutor
    :param executor:
    :return:
    """
    executor.run_function_n_times()
    results = executor.analyse_result()
    executor.reset_container()

    return results


if __name__ != "__main__":
    pass
else:
    max_not_improve = 10
    time_segments_iter = 5
    repeat_times = 1
    gama = 0.24
    json_modifier = JsonModifier("../data/configure.json")
    sat_number = [1, 2, 3, 4, 5, 6]
    task_number = [200, 400, 500, 600, 800, 1000, 1200, 1500]
    Scenario = [(item1, item2) for item1 in sat_number for item2 in task_number]
    rng = np.random.default_rng(3845)
    rng.shuffle(Scenario)
    alpha = 0.2
    stop_degree = 0.001

    executor_alns_cnp = Executor(repeat_times, execute_pre_tuned_alns_cnp, gama, max_not_improve, time_segments_iter)

    # evaluation system: benefits  runtimes total_benefits
    # create repeater and execute it for appointed times
    update_interval = 11
    exit_flag = False
    a = 1
    error_curve = []
    i = 0
    while True:
        for scenario in Scenario:
            i += 1
            print("训练轮数：第{}轮".format(i))

            with open('../data/weight_trained.pkl', 'rb') as f:
                weight_matrix_initial = pickle.load(f)

            json_modifier.modify_and_save({
                "Task_Number": scenario[1],
                "Satellite_Number": scenario[0]
            }
            )
            # train
            weights_trained = run_and_analyse(executor_alns_cnp)
            # grads
            delta_weight_matrix = weights_trained - weight_matrix_initial
            # update
            weight_matrix_eventual = weight_matrix_initial + alpha * delta_weight_matrix
            # stop criterion
            vec1 = weight_matrix_eventual.flatten()
            vec2 = weight_matrix_initial.flatten()
            update_degree = chebyshev(vec1, vec2)
            error_curve.append(update_degree)
            print("update degree = {}".format(update_degree))

            with open('../data/weight_trained.pkl', 'wb') as f:
                pickle.dump(weight_matrix_eventual, f)

            if len(error_curve) % 50 == 0:
                alpha *= 0.5

            if len(error_curve) >= 200:
                print(weights_trained)
                exit_flag = True
                break
        if exit_flag:
            break
    with open('../data/weight_trained_curve.pkl', 'wb') as f:
        pickle.dump(error_curve, f)

