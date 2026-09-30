import matplotlib.pyplot as plt
import scienceplots
import pickle
from pprint import pprint
from dataclasses import dataclass, field
import numpy as np

plt.style.use(['science', 'ieee'])

@dataclass
class AlgorithmResult(object):
    mean_time: list = field(default_factory=list)
    std_time: list = field(default_factory=list)
    mean_benefit: list = field(default_factory=list)
    std_benefit: list = field(default_factory=list)
    total_benefit: list = field(default_factory=list)


class Analyser(object):
    data_reshaped: dict

    def __init__(self):
        self.data_reshaped = {}
        self.color_palette = {"pre_tuned_alns_cnp": '#E0AC04',
                              "SA": '#0223E0',
                              "GA": '#10E03D',
                              "ACS": '#E11206',
                              }

    def read_and_reorganize(self):
        # result_compare_with_central_scale_sat_num.pkl
        # result_compare_with_central_scale_task.pkl
        # result_compare_with_central_scale_task_pack_pnac.pkl
        # result_compare_with_central_scale_task_pack_ga.pkl
        with open('../data/result_compare_with_central_scale_sat_num.pkl', 'rb') as f:
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

    def execute(self):
        self.read_and_reorganize()
        pprint(self.data_reshaped)

        n_groups = len(self.data_reshaped["ACS"].mean_time)
        print(n_groups)
        index = np.arange(n_groups) * 1.15
        bar_width = 0.25

        error_params = {"elinewidth": 1,
                        "ecolor": 'black',
                        "capsize": 5,
                        "capthick": 1
                        }
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))

        for sub_index, key in enumerate(self.data_reshaped):
            std_err = np.array(self.data_reshaped[key].std_benefit) / np.array(self.data_reshaped[key].total_benefit)
            # 绘制三组数据
            bar = ax.bar(index + sub_index * bar_width, np.array(self.data_reshaped[key].mean_benefit) /
                         np.array(self.data_reshaped[key].total_benefit),
                         bar_width,
                         color=self.color_palette[key],
                         yerr=std_err,
                         error_kw=error_params,
                         alpha=0.6,
                         label=key)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('Number of Tasks', fontsize=10, labelpad=2)
        ax.set_ylabel('Benefit Completion Rate', fontsize=10, labelpad=2)
        ax.legend(fontsize=10, ncol=1)

        # 设置X轴的刻度标签
        ax.set_xticks(index + bar_width)  # 将标签设置在组的中间
        ax.set_xticklabels([f'C {i}' for i in range(1, n_groups + 1)])

        # 显示图形
        plt.show()


if __name__ != "__main__":
    pass
else:
    analyser = Analyser()
    analyser.execute()



