"""
compare_with_ga_acs_sa
"""
#!/usr/bin/env python3
from algorithms.ELH_ACNP import execute_pre_tuned_alns_cnp
from algorithms.GA import execute_ga
from algorithms.ACS import execute_acs
from algorithms.SA import execute_sa
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import json
import pickle

plt.style.use(['science', 'ieee'])

class Executor:
    """
    Experiment Repeater
    """
    runtimes: list
    benefits: list
    total_benefits: list

    def __init__(self, n, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.execute_times = n
        self.runtimes = []
        self.benefits = []
        self.total_benefits = []

    def run_function_n_times(self):
        """
        run experiment repeat n times
        :return:
        """
        for _ in range(self.execute_times):
            try:
                runtime, benefit, total_benefit = self.func(*self.args, **self.kwargs)
            except:
                runtime, benefit, total_benefit = np.nan, np.nan, np.nan
            self.runtimes.append(runtime)
            self.benefits.append(benefit)
            self.total_benefits.append(total_benefit)

    def analyse_result(self):
        """
        analyse the result
        :return:
        """
        mean_run_time = np.mean(self.runtimes)
        variance_run_time = np.std(self.runtimes)
        mean_benefits = np.mean(self.benefits)
        variance_benefits = np.std(self.benefits)
        return mean_run_time, variance_run_time, mean_benefits, variance_benefits, self.total_benefits[0]

    def reset_container(self):
        """
        reset result container
        :return:
        """
        self.runtimes = []
        self.benefits = []
        self.total_benefits = []

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
    # set experiment parameter
    start, end, step = 1, 7, 1
    repeat_times = 5
    gama = 0.24
    json_modifier = JsonModifier("../data/configure.json")

    executor_pre_tuned_alns_cnp = Executor(repeat_times, execute_pre_tuned_alns_cnp, gama)
    executor_ga = Executor(repeat_times, execute_ga)
    executor_sa = Executor(repeat_times, execute_sa)
    executor_acs = Executor(repeat_times, execute_acs)

    Result = {
        "pre_tuned_alns_cnp": [],
        "GA": [],
        "SA": [],
        "ACS": []
    }

    # evaluation system: benefits  runtimes total_benefits
    # create repeater and execute it for appointed times

    for sat_num in np.arange(start, end, step):
        json_modifier.modify_and_save({
            "Task_Number": 600,
            "Satellite_Number": int(sat_num)
        }
        )

        result_pre_tuned_alns_cnp = run_and_analyse(executor_pre_tuned_alns_cnp)
        Result["pre_tuned_alns_cnp"].append(result_pre_tuned_alns_cnp)

        result_ga = run_and_analyse(executor_ga)
        Result["GA"].append(result_ga)

        result_sa = run_and_analyse(executor_sa)
        Result["SA"].append(result_sa)

        result_acs = run_and_analyse(executor_acs)
        Result["ACS"].append(result_acs)

    with open('../data/result_compare_with_central_scale_sat_num.pkl', 'wb') as f:
        pickle.dump(Result, f)

