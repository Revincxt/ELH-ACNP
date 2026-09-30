import pickle
from dataclasses import dataclass, field
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

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

class Painter(object):
    data_reshaped: dict
    start: int
    end: int
    step: int

    def __init__(self):
        self.data_reshaped = {}
        self.start = 50
        self.end = 1501
        self.step = 10
        self.color_palette = {"pre_tuned_alns_cnp": '#FA0B00',  # BORDEAUX
                              "alns_cnp": '#0100FA',  # BURGUNDY
                              "lns_cnp": '#FACE05',  # CHINA RED
                              "vns_cnp": '#05FA62',  # KLEIN BLUE
                              "vnd_cnp": '#A5923A',  # PRUSSIAN BLUE
                              "ns_cnp": '#7A4340',  # TIFFANY BLUE
                              "hpfs_cnp": '#40407A',  # MARS GREEN
                              "fcfs_cnp": '#407A56',  # SENNELIER YELLOW
                              "llf_cnp": '#73EA8B'  # HERMES ORANGE
                              }
        self.new_labels = {"pre_tuned_alns_cnp": 'ELH-ACNP',
                           "alns_cnp": 'EH-ACNP',
                           "lns_cnp": 'LNS-ICNP',
                           "vns_cnp": 'VNS-ICNP',
                           "vnd_cnp": 'VND-ICNP',
                           "ns_cnp": 'H-ICNP',
                           "hpfs_cnp": 'HPFS-ICNP',
                           "fcfs_cnp": 'FCFS-ICNP',
                           "llf_cnp": 'LLF-ICNP'
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

    def draw_benefit_mix_mean_error_graph(self):
        """
        draw the benefit graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(nrows=2, dpi=200, figsize=(8, 6))
        ax[0].set_xlim([45, 1505])
        # ax[0].set_ylim([0.75, 1])
        ax[1].set_xlim([45, 1505])
        # ax[1].set_ylim([0.75, 1])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            if key == "pre_tuned_alns_cnp" or key == "alns_cnp":
                ax[0].plot(x_values, np.array(self.data_reshaped[key].mean_benefit) /
                           np.array(self.data_reshaped[key].total_benefit),
                           color=self.color_palette[key],
                           linestyle="solid",
                           label=self.new_labels[key],
                           linewidth=1)
                y_low = (np.array(self.data_reshaped[key].mean_benefit) - np.array(self.data_reshaped[key].std_benefit))/\
                        np.array(self.data_reshaped[key].total_benefit)
                y_high = (np.array(self.data_reshaped[key].mean_benefit) + np.array(self.data_reshaped[key].std_benefit))/\
                         np.array(self.data_reshaped[key].total_benefit)
                ax[0].fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.35)

                ax[1].plot(x_values, self.data_reshaped[key].mean_time,
                           color=self.color_palette[key],
                           linestyle="solid",
                           label=self.new_labels[key],
                           linewidth=1)

                y_low = np.array(self.data_reshaped[key].mean_time) - np.array(self.data_reshaped[key].std_time)
                y_high = np.array(self.data_reshaped[key].mean_time) + np.array(self.data_reshaped[key].std_time)
                ax[1].fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.35)
            else:
                continue
        ax[0].tick_params(axis='both', which='major', labelsize=14)
        ax[0].set_xlabel('Number of Tasks', fontsize=14, labelpad=2)
        ax[0].set_ylabel('Benefit Completion Rate', fontsize=14, labelpad=2)
        ax[0].legend(fontsize=14, ncol=2)
        ax[1].tick_params(axis='both', which='major', labelsize=14)
        ax[1].set_xlabel('Number of Tasks', fontsize=14, labelpad=2)
        ax[1].set_ylabel('Algorithm Running Time(s)', fontsize=14, labelpad=2)
        ax[1].legend(fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig("pre_not_sat_num.pdf")
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
        # ax.set_xlim([50, 1500])
        # ax.set_ylim([0, 40])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            if key == "pre_tuned_alns_cnp" or key == "alns_cnp":
                ax.plot(x_values, self.data_reshaped[key].mean_time,
                        color=self.color_palette[key],
                        linestyle="solid",
                        label=self.new_labels[key],
                        linewidth=1)

                y_low = np.array(self.data_reshaped[key].mean_time) - np.array(self.data_reshaped[key].std_time)
                y_high = np.array(self.data_reshaped[key].mean_time) + np.array(self.data_reshaped[key].std_time)
                ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.3)
            else:
                continue

        plt.show()

    def run(self):
        self.read_and_reorganise_data()
        self.draw_benefit_mix_mean_error_graph()
        # self.draw_runtime_mix_mean_error_graph()


if __name__ == "__main__":
    painter = Painter()
    painter.run()
