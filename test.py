import pickle
from pprint import pprint
from dataclasses import dataclass, field
import numpy as np

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

    def read_and_reorganize(self):
        # result_compare_with_central_scale_sat_num.pkl
        # result_compare_with_central_scale_task.pkl
        # result_compare_with_central_scale_task_pack_pnac.pkl
        # result_compare_with_central_scale_task_pack_ga.pkl
        with open('./data/result_compare_with_central_scale_task_pack_ga.pkl', 'rb') as f:
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
        for key in self.data_reshaped:
            # print("{}收益完成率 稳定性:".format(key))
            # print(np.array(self.data_reshaped[key].mean_benefit) / np.array(self.data_reshaped[key].total_benefit))
            # print(np.array(self.data_reshaped[key].std_benefit) / np.array(self.data_reshaped[key].total_benefit))

            print("{}   CPU耗时 稳定性:".format(key))
            print(self.data_reshaped[key].mean_time)
            print(self.data_reshaped[key].std_time)


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
