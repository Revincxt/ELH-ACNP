import pickle
from pprint import pprint
from dataclasses import dataclass, field
import scienceplots
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'ieee'])

# from matplotlib.font_manager import FontProperties
# # font = FontProperties(fname=r"C:\Windows\Fonts\SimSun.ttc")
# font = FontProperties(family="SimSun", size=14)
# plt.style.use(['science', "no-latex"])

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
        self.start = 1
        self.end = 7
        self.step = 1
        self.marker_palette = {"pre_tuned_alns_cnp": "s",
                               "GA": "*",
                               "SA": "h",
                               "ACS": "D"
                               }
        self.color_palette = {"pre_tuned_alns_cnp": '#FA0B00',  # BORDEAUX
                              "alns_cnp": '#0100FA',  # BURGUNDY
                              "lns_cnp": '#FACE05',  # CHINA RED
                              "vns_cnp": '#05FA62',  # KLEIN BLUE
                              "vnd_cnp": '#A5923A',  # PRUSSIAN BLUE
                              "ns_cnp": '#7A4340',  # TIFFANY BLUE
                              "hpfs_cnp": '#40407A',  # MARS GREEN
                              "fcfs_cnp": '#407A56',  # SENNELIER YELLOW
                              "llf_cnp": '#73EA8B',  # HERMES ORANGE
                              "GA": '#5507F3',  # BURGUNDY
                              "SA": '#362181',  # CHINA RED
                              "ACS": '#2E89C3',  # KLEIN BLUE
                              }
        self.new_labels = {"pre_tuned_alns_cnp": 'ELH-ACNP',
                           "alns_cnp": 'EH-ACNP',
                           "lns_cnp": 'LNS-ICNP',
                           "vns_cnp": 'VNS-ICNP',
                           "vnd_cnp": 'VND-ICNP',
                           "ns_cnp": 'H-ICNP',
                           "hpfs_cnp": 'HPFS-ICNP',
                           "fcfs_cnp": 'FCFS-ICNP',
                           "llf_cnp": 'LLF-ICNP',
                           "GA": "GA-ELUMS",
                           "SA": "ISA",
                           "ACS": "EHE-DCF"
                           }

    def read_and_reorganize(self):
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

    @staticmethod
    def log10(x):
        x += np.array([1])
        return np.log(x) / np.log(10)

    def draw_benefit_mix_mean_error_graph(self):
        """
        draw the benefit graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        ax.set_xlim([0.9, 6.1])
        # ax.set_ylim([0.48, 1.01])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            ax.errorbar(x_values, np.array(self.data_reshaped[key].mean_benefit) /
                        np.array(self.data_reshaped[key].total_benefit),
                        xerr=None,
                        yerr=np.array(self.data_reshaped[key].std_benefit) / np.array(
                            self.data_reshaped[key].total_benefit),
                        linestyle="solid",
                        color=self.color_palette[key],
                        linewidth=2,
                        ecolor=self.color_palette[key],
                        elinewidth=2,
                        capsize=5,
                        capthick=2,
                        marker='o',
                        markersize=10,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        markerfacecolor=self.color_palette[key],
                        label=self.new_labels[key]
                        )
            # ax.plot(x_values, np.array(self.data_reshaped[key].mean_benefit) /
            #         np.array(self.data_reshaped[key].total_benefit),
            #         marker=self.marker_palette[key],
            #         color=self.color_palette[key],
            #         linestyle="solid",
            #         label=self.new_labels[key],
            #         linewidth=1)
            y_low = (np.array(self.data_reshaped[key].mean_benefit) - np.array(self.data_reshaped[key].std_benefit))/\
                    np.array(self.data_reshaped[key].total_benefit)
            y_high = (np.array(self.data_reshaped[key].mean_benefit) + np.array(self.data_reshaped[key].std_benefit))/\
                     np.array(self.data_reshaped[key].total_benefit)
            ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.35)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Number of Satellites', fontsize=14, labelpad=2)
        ax.set_ylabel('Benefit Completion Rate', fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig("offline_sat_num_bcr.pdf")
        plt.show()

        # # Add labels and title

        # ax.set_title('Algorithm Convergence Curve', fontsize=8)
        # ax.tick_params(axis='x', labelsize=8)
        # ax.tick_params(axis='y', labelsize=8)
        #
        # # Customize the plot
        # plt.grid(False)
        # # plt.ylim(0, 1)
        #
        # # Show the plot
        # plt.show()

    def draw_runtime_mix_mean_error_graph(self):
        """
        draw the runtime graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        ax.set_xlim([0.9, 6.1])
        # #ax.set_ylim([0, 40])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            ax.errorbar(x_values, np.array(self.data_reshaped[key].mean_time),
                        xerr=None,
                        yerr=np.array(self.data_reshaped[key].std_time),
                        linestyle="solid",
                        color=self.color_palette[key],
                        linewidth=2,
                        ecolor=self.color_palette[key],
                        elinewidth=2,
                        capsize=5,
                        capthick=2,
                        marker='o',
                        markersize=10,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        markerfacecolor=self.color_palette[key],
                        label=self.new_labels[key]
                        )
            # ax.plot(x_values, self.data_reshaped[key].mean_time,
            #         marker=self.marker_palette[key],
            #         color=self.color_palette[key],
            #         linestyle="solid",
            #         label=self.new_labels[key],
            #         linewidth=1
            #         )
            y_low = np.array(self.data_reshaped[key].mean_time) - np.array(self.data_reshaped[key].std_time)
            y_high = np.array(self.data_reshaped[key].mean_time) + np.array(self.data_reshaped[key].std_time)
            ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.35)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Number of Satellites', fontsize=14, labelpad=2)
        ax.set_ylabel('Algorithm Running Time(s)', fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig("offline_sat_num_runtime.pdf")
        plt.show()

    def execute(self):
        self.read_and_reorganize()
        self.draw_benefit_mix_mean_error_graph()
        self.draw_runtime_mix_mean_error_graph()
        # pprint(self.data_reshaped)
        # for key in self.data_reshaped:
        #     print("{}收益完成率 稳定性:".format(key))
        #     print(np.array(self.data_reshaped[key].mean_benefit) / np.array(self.data_reshaped[key].total_benefit))
        #     print(np.array(self.data_reshaped[key].std_benefit) / np.array(self.data_reshaped[key].total_benefit))
        #
            # print("{}   CPU耗时 稳定性:".format(key))
            # print(self.data_reshaped[key].mean_time)
            # print(self.data_reshaped[key].std_time)


if __name__ != "__main__":
    pass
else:
    analyser = Analyser()
    analyser.execute()

# pprint(data)
# for key in data:
#     print(len(data[key]))
#    # print(result)
#     # print(len(result))

# print(os.cpu_count())
