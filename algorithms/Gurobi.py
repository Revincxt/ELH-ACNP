from task_tool.Task_generator import Task
from task_tool.HelpLibrary import Helper
from lib.Agent import Satellite
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB, LinExpr
import re


plt.style.use(['science', 'ieee'])

class Gu(object):
    bandwidth: int
    task_num: int
    satellite_num: int
    tasklist: np.ndarray
    task_indices: set
    conflict_list: list
    satellite_list: list[Satellite]
    execution_list: list
    time_list: list
    scene_info: list
    E: float

    def __init__(self, config_dir):
        # world info
        self.config_dir = config_dir
        self.X = {}
        self.T = {}
        self.P = {}
        self.Q = {}
        self.model = gp.Model("HTS_Sat_S")
        self.model.setParam('Method', 3)
        self.M = 1e6
        self.S = 0
        self.Z = {}

    def initial_task_world(self):
        """
        初始化任务, 赋值类属性：task_num, satellite_num, bandwidth, conflict_set, tasklist
        task_info:[task_num,
                  distribute_mean_1,
                  distribute_std_1,
                  distribute_mean_2,
                  distribute_std_2
                 ]
        scene_info:[start_time,
                   end_time
                   ]
        resource_info:[satellite_num,
                      unit_bandwidth,
                      max_frequency,
                      min_frequency
                     ]
        :return: None
        """
        # 读取文件
        file_helper = Helper(self.config_dir)
        task_info, self.scene_info, resource_info = file_helper.parse_world_info()
        self.task_num, distribute_mean_1, distribute_std_1, distribute_mean_2, distribute_std_2 = task_info
        self.satellite_num, _, _, self.bandwidth = resource_info
        self.E = self.scene_info[1] - self.scene_info[0]
        # 生成任务
        task = Task(self.task_num,
                    self.scene_info[0], self.scene_info[1],
                    distribute_mean_1, distribute_std_1,
                    distribute_mean_2, distribute_std_2
                    )
        self.tasklist = task.tasklist

        self.time_list = [[-1] * self.task_num for _ in range(self.satellite_num)]
        self.execution_list = [[-1] * self.task_num for _ in range(self.satellite_num)]

    def define_variable(self):
        """
        添加变量
        :return:
        """
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    name = "X_" + str(k) + "_" + str(i) + "_" + str(j)
                    self.X[k, i, j] = self.model.addVar(0,
                                                        1,
                                                        vtype=GRB.BINARY,
                                                        name=name)
                    name = "p_" + str(k) + "_" + str(i) + "_" + str(j)
                    self.P[k, i, j] = self.model.addVar(0,
                                                        1,
                                                        vtype=GRB.BINARY,
                                                        name=name)
                    name = "q_" + str(k) + "_" + str(i) + "_" + str(j)
                    self.Q[k, i, j] = self.model.addVar(0,
                                                        1,
                                                        vtype=GRB.BINARY,
                                                        name=name)
                name = "T_" + str(k) + "_" + str(i)
                self.T[k, i] = self.model.addVar(lb=self.S,
                                                 ub=self.E,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=name)

        self.model.update()

    def add_target(self):
        """
        添加目标
        :return:
        """
        # obj = gp.QuadExpr()
        obj = LinExpr(0)
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                # obj.addTerms(self.tasklist[i, 4], (self.tasklist[i, 1]-self.X[k, i, 0])/(self.tasklist[i, 1] - se
                # lf.tasklist[i,0]), self.X[k, i, 0])
                obj.addTerms(self.tasklist[i, 4],  self.X[k, i, i])
        self.model.setObjective(obj, GRB.MAXIMIZE)
        self.model.update()

    def add_constraints(self):
        """
        添加约束
        :return:
        """
        # 1. 添加约束1
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    self.model.addConstr(self.X[k, i, i] >= self.X[k, i, j], name="constraint1.2")
        # 2. 添加约束2
        for i in range(self.task_num):
            for j in range(self.task_num):
                lhs = LinExpr(0)
                for k in range(self.satellite_num):
                    lhs.addTerms(1, self.X[k, i, j])
                self.model.addConstr(lhs <= 1, name="onetime2")
        # 3. 添加约束3
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                lhs = LinExpr(0)
                for _k in range(self.satellite_num):
                    for j in range(self.task_num):
                        if _k != k:
                            lhs.addTerms(1, self.X[_k, i, j])
                self.model.addConstr(lhs <= self.M * (1 - self.X[k, i, i]), name="strict3")
        # 4. 添加约束4
        for i in range(self.task_num):
            for k in range(self.satellite_num):
                self.model.addConstr(self.T[k, i] + self.M * (1 - self.X[k, i, i]) >= self.tasklist[i, 0], name="strict4")

        # 5. 添加约束5
        for i in range(self.task_num):
            for k in range(self.satellite_num):
                self.model.addConstr(self.T[k, i] + self.tasklist[i, 3]
                                     * self.X[k, i, i] - self.M * (1 - self.X[k, i, i]) <= self.tasklist[i, 1], name="strict5")

        # # 6. 添加约束6
        # for i in range(self.task_num):
        #     for k in range(self.satellite_num):
        #         self.model.addConstr(self.T[k, i] + self.M * (1 - self.X[k, i, i]) >= self.S,
        #                              name="strict6")
        #
        # # 7. 添加约束7
        # for i in range(self.task_num):
        #     for k in range(self.satellite_num):
        #         self.model.addConstr(self.T[k, i] + self.tasklist[i, 3] * self.X[k, i, i] - self.M * (1 -
        #                                                                                               self.X[
        #                                                                                                   k, i, i]) <=
        #                              self.E, name="strict7")

        # 8 约束8
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    self.model.addConstr(self.T[k, i] - self.T[k, j] <= self.M * (1 - self.P[k, i, j]), name="strict8")

        # 9 约束9
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    self.model.addConstr(self.T[k, j] + self.tasklist[j, 3] - self.T[k, i] <= self.M * (1 - self.Q[k, i, j]), name="strict9")
        # 10 约束10
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    self.model.addConstr(self.X[k, i, j] >= self.P[k, i, j] + self.Q[k, i, j] - 1, name="strict10")
        # 11 约束11
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    self.model.addConstr(self.X[k, i, j] - self.P[k, i, j] <= 0, name="strict11")
        # 12 约束12
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                for j in range(self.task_num):
                    self.model.addConstr(self.X[k, i, j] - self.Q[k, i, j] <= 0, name="strict12")
        # 13. 添加约束13
        for k in range(self.satellite_num):
            for i in range(self.task_num):
                lhs = LinExpr(0)
                for j in range(self.task_num):
                    lhs.addTerms(self.tasklist[j, 2], self.X[k, i, j])
                self.model.addConstr(lhs <= self.bandwidth, name="strict13")

        # 13. 添加约束11
        for i in range(self.task_num):
            for k in range(self.satellite_num):
                self.model.addConstr(self.T[k, i] + self.tasklist[i, 3] <= self.E, name="strict8")
        self.model.update()

    def print_variable(self):
        """
        打印变量
        :return:
        """
        print("变量取值", "-*-" * 10)
        # 打印出所有整数变量（包括二进制变量）的求解结果
        for v in self.model.getVars():
            if v.vtype in [GRB.INTEGER, GRB.BINARY] and v.varName.split('_')[2] == v.varName.split('_')[3]:
                self.execution_list[int(v.varName.split('_')[1])][int(v.varName.split('_')[2])] = v.x
                print('整数变量 %s 的求解结果是 %g' % (v.varName, v.x), int(v.varName.split('_')[1]) , int(v.varName.split('_')[2]))

        for v in self.model.getVars():
            if v.vtype == GRB.CONTINUOUS:
                self.time_list[int(v.varName.split('_')[1])][int(v.varName.split('_')[2])] = v.x
                print('连续变量 %s 的求解结果是 %g' % (v.varName, v.x), int(v.varName.split('_')[1]), int(v.varName.split('_')[2]))
        print(self.time_list)

    def run(self):
        """
        执行优化
        :return:
        """
        self.initial_task_world()
        self.define_variable()
        self.add_target()
        self.add_constraints()
        self.model.optimize()
        if self.model.status == GRB.Status.OPTIMAL or self.model.status == GRB.Status.TIME_LIMIT:
            print("obj = {0}".format(self.model.ObjVal))
        # self.print_variable()
        # self.plot_schedule()
        # print("执行列表", self.execution_list[0])
        return self.model.ObjVal
    #
    # def plot_schedule(self):
    #     """
    #     绘制资源占用图
    #     :return:
    #     """
    #     # plot agent schedules
    #     fig_schedule = plt.figure(1)
    #     fig_schedule.suptitle("Spectrum Resource Usage", fontsize=6, x=0.5, y=0.95, ha='center', va='top')
    #     fig_schedule.text(0.06, 0.5, 'Bandwidth', va='center', rotation='vertical', fontsize=6)
    #     for satellite in range(self.satellite_num):
    #         ax = plt.subplot(self.satellite_num, 1, satellite + 1)
    #         ax.set_title("Satellite " + str(satellite), fontsize=6, pad=3)
    #         ax.tick_params(axis='both', which='major', labelsize=6)
    #         if satellite == (self.satellite_num - 1):
    #             ax.set_xlabel("Time", fontsize=6)
    #         ax.set_xlim([0, 0.5])  # 时间范围
    #         # ax.set_ylim([0, 22])
    #
    #         # 设置颜色
    #         # colors = ['red', 'green', 'blue', 'orange', 'yellow', 'pink', 'purple', 'black', 'gray']
    #         events = self.converse_task_to_events(satellite)
    #
    #         curr_width = 0
    #         last_event_time = 0
    #         for time, width_change in events:
    #             if time > last_event_time:
    #                 length = time - last_event_time
    #                 rect = patches.Rectangle((last_event_time, 0), time - last_event_time, curr_width,
    #                                          facecolor='lightpink')
    #                 ax.add_patch(rect)
    #                 # 绘图
    #             last_event_time = time
    #             curr_width += width_change
    #
    #     fig_schedule.subplots_adjust(hspace=0.5)
    #
    #     # set legends
    #     colors = ["red", "red"]
    #     line_styles = ["-", "-."]
    #     line_width_list = [10, 2]
    #     labels = ["Assignment Time", "Task Time"]
    #
    #     def f(line_style, color_type, line_width):
    #         return plt.plot([], [], linestyle=line_style, color=color_type,
    #                         linewidth=line_width)[0]
    #
    #     handles = [f(line_styles[i], colors[i], line_width_list[i]) for i in range(len(labels))]
    #     fig_schedule.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)
    #
    #     plt.show()
    #
    # def converse_task_to_events(self, sat_index):
    #     """
    #     卫星任务集转事件集
    #     :param sat_index:
    #     :return: events
    #     """
    #     occupied_windows = [[self.time_list[sat_index][index],
    #                          self.tasklist[index, 3],
    #                          self.tasklist[index, 2]]
    #                         for index, value in enumerate(self.execution_list[sat_index]) if value == 1
    #                         ]
    #     events = []
    #     # Add start and end events
    #     # Add occupied_time_window events
    #     for occupied_start, duration, occupied_width in occupied_windows:
    #         events.append((occupied_start, occupied_width))
    #         events.append((occupied_start + duration, -occupied_width))
    #     # Sort by time
    #     events.sort()
    #     return events


if __name__ == "__main__":
    config_file_name = "../data/configure.json"

    print("-----VRP_Gurobi------")
    start_time2 = time.time()  # 记录模型开始时间
    Algorithm = Gu(config_dir=config_file_name)
    result_gurobi = Algorithm.run()
    end_time2 = time.time()
    # print("cplex_model  运行结果={0}  求解时间={1:<.3} 秒".format(result_cplex,end_time - start_time))
    print("gurobi_model 运行结果={0}  求解时间={1:<.3} 秒".format(result_gurobi, end_time2 - start_time2))
    # aa = 0
    # for i in Algorithm.tasklist:
    #     aa += i[4]
    # print(aa)














