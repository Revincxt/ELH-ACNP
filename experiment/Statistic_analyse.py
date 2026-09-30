import pickle
from dataclasses import dataclass, field
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

plt.style.use(['science', 'ieee'])


@dataclass
class AlgorithmResult(object):
    mean_time: list = field(default_factory=list)
    std_time: list = field(default_factory=list)
    mean_benefit: list = field(default_factory=list)
    std_benefit: list = field(default_factory=list)
    total_benefit: list = field(default_factory=list)


class Analyst(object):
    data_reshaped: dict
    start: int
    end: int
    step: int

    def __init__(self):
        self.data_reshaped = {}
        self.start = 50
        self.end = 1501
        self.step = 10
        self.color_palette = {"pre_tuned_alns_cnp": 'green',  # BORDEAUX
                              "alns_cnp": '#800020',  # BURGUNDY
                              "lns_cnp": '#B05923',  # CHINA RED
                              "vns_cnp": '#002FA7',  # KLEIN BLUE
                              "vnd_cnp": '#003153',  # PRUSSIAN BLUE
                              "ns_cnp": '#81D8D0',  # TIFFANY BLUE
                              "hpfs_cnp": '#008C8C',  # MARS GREEN
                              "fcfs_cnp": '#F9DC24',  # SENNELIER YELLOW
                              "llf_cnp": '#E85827'  # HERMES ORANGE
                              }

    def read_and_reorganise_data(self):
        """
        read the running results and reorganize it
        :return:
        """
        with open('../data/result_scale_task.pkl', 'rb') as f:
            results = pickle.load(f)
        navigator = list(results.keys())[0]
        for key in iter(results):
            self.data_reshaped[key] = AlgorithmResult()
        for index in range(len(results[navigator])):
            for key in iter(results):
                self.data_reshaped[key].mean_time.append(results[key][index][0])
                self.data_reshaped[key].std_time.append(results[key][index][1])
                self.data_reshaped[key].mean_benefit.append(results[key][index][2])
                self.data_reshaped[key].std_benefit.append(results[key][index][3])
                self.data_reshaped[key].total_benefit.append(results[key][index][4])

    def statistic_analysis(self):
        """
        execute a statistic analysis between pre_trained and raw ALNS_CNP algorithm
        :return:
        """
        runtime_trained = self.data_reshaped["pre_tuned_alns_cnp"].mean_time
        runtime_raw = self.data_reshaped["alns_cnp"].mean_time

        benefit_trained = self.data_reshaped["pre_tuned_alns_cnp"].mean_benefit
        benefit_raw = self.data_reshaped["alns_cnp"].mean_benefit

        runtime_advance_percentage = np.mean(np.array(runtime_raw) - np.array(runtime_trained)) / np.mean(np.array(runtime_raw))
        print(np.mean(np.array(runtime_raw)))
        print("算法速度提升百分比: {:.2%}".format(runtime_advance_percentage))

        benefit_advance_percentage = np.mean(np.array(benefit_trained) - np.array(benefit_raw))
        print("算法收益提升", benefit_advance_percentage)

    def run(self):
        """
        entrance of program
        :return:
        """
        self.read_and_reorganise_data()
        self.statistic_analysis()


if __name__ == "__main__":
    analyst = Analyst()
    analyst.run()

