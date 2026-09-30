#!/usr/bin/env python3
from task_tool.Task_generator import Task
from task_tool.HelpLibrary import Helper
from lib.Agent import Satellite
from lib.ACS import Edge, Ant
from typing import List, Optional
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import numpy as np
from tqdm import trange
import time
import os


plt.style.use(['science', 'ieee'])

class ACS(object):
    bandwidth: int
    task_num: int
    satellite_num: int
    tasklist: np.ndarray
    task_indices: set
    conflict_list: list
    satellite_list: list[Satellite]

    # main result
    score_list: list  # 2D list
    time_list: list  # 2D list
    frequency_list: list  # 2D list

    # algorithm parameter
    edges: list[list]  # virtual task graph
    ants: list[Ant]  # colony
    elite: dict

    # iteration curve
    benefit_curve: list
    time: float

    def __init__(self, config_dir, colony_size=10, alpha=3.0, beta=1.0, rho=0.10, pheromone_deposit_weight=0.01,
                 initial_pheromone=1.0, steps=100):
        # world info
        self.config_dir = config_dir
        # algorithm parameter
        self.colony_size = colony_size
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.initial_pheromone = initial_pheromone

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
        task_info, scene_info, resource_info = file_helper.parse_world_info()
        self.task_num, distribute_mean_1, distribute_std_1, distribute_mean_2, distribute_std_2 = task_info
        self.satellite_num, _, _, self.bandwidth = resource_info
        # 生成任务
        task = Task(self.task_num,
                    scene_info[0], scene_info[1],
                    distribute_mean_1, distribute_std_1,
                    distribute_mean_2, distribute_std_2
                    )
        # 生成卫星集
        self.tasklist = task.tasklist
        self.conflict_list = task.conflict_set
        self.task_indices = set(range(self.task_num))

        # 初始化参数
        self.satellite_list = [Satellite(self.bandwidth, i) for i in range(self.satellite_num)]
        self.time_list = [[-1] * self.task_num for _ in range(self.satellite_num)]

        # 初始化状态
        self.benefit_curve = []
        self.elite = {"fitness": 0,
                      "tour": 0
                      }

    def assign_tasks(self, task_sequences):
        """
        根据输入序列分配任务
        :param task_sequences:
        :return: score
        """
        # 1. set the score container
        score = 0
        # 2. clear the history information
        for satellite in iter(self.satellite_list):
            satellite.clear_execution_list()
        self.time_list = [[-1] * self.task_num for _ in range(self.satellite_num)]
        # 3. compute
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.satellite_num) as executor:
            for index_iter in iter(task_sequences):

                # 1. 找出任务冲突集
                potential_conflict_list = self.conflict_list[index_iter]

                # 2. 提取任务信息
                task_start_time, task_end_time, task_width, task_serve_time, task_priority, _ = self.tasklist[
                    index_iter]

                # 3. 构建申请窗口
                request_time_window = [task_start_time, task_end_time]

                # 4. 构建参数包
                parameter_package = [[self.bandwidth, request_time_window, task_serve_time, task_width]
                                     for _ in self.satellite_list
                                     ]
                # 5. 补充冲突任务信息
                for satellite in iter(self.satellite_list):
                    # 5.1 找出执行集中的冲突任务
                    planned_list = satellite.execution_list
                    conflict_set = potential_conflict_list & planned_list

                    # 5.2 提取冲突信息
                    occupied_windows = [[self.time_list[satellite.satellite_id][i],
                                         self.tasklist[i, 3],
                                         self.tasklist[i, 2]]
                                        for i in conflict_set
                                        ]
                    # 5.3 添加参数
                    parameter_package[satellite.satellite_id].append(occupied_windows)
                    parameter_package[satellite.satellite_id].append(satellite.satellite_id)

                # 6. 计算可用时间窗<进程池>
                futures = {executor.submit(self.find_time_slices, packet)
                           for packet in iter(parameter_package)
                           }
                concurrent.futures.wait(futures)
                # 7. 裁决任务
                earliest_time = float('inf')
                best_frequency = None
                best_satellite = None
                for future in concurrent.futures.as_completed(futures):

                    if future.result() and future.result()[0] < earliest_time:
                        earliest_time = future.result()[0]
                        best_frequency = future.result()[1]
                        best_satellite = future.result()[2]

                if best_frequency:
                    # 获取得到了最佳执行时间：best_scheme--[time, band]
                    # 最佳卫星：best_satellite
                    # todo：记录 1.satellite_list 集合添加任务 2.time_list添加时间 3.收益记录：score_list 4.频点记录：frequency_list
                    self.satellite_list[best_satellite].add_task(index_iter)
                    self.time_list[best_satellite][index_iter] = earliest_time
                    score += self.tasklist[index_iter, 4]
        return score

    @staticmethod
    def find_time_slices(args):
        """
        寻找时间窗<找到第一个窗口直接返回，不找出全部的>
        :param args:
                -bandwidth:
                -requested_time_window:
                -slice_length:
                -slice_width:
                -occupied_time_windows:
        :return: resource information or None
        """
        # unpack the parameter package
        bandwidth, requested_time_window, slice_length, slice_width, occupied_time_windows, satellite_id = args

        if not occupied_time_windows:
            return [requested_time_window[0], bandwidth, satellite_id]

        start_time, end_time = requested_time_window

        events = [(start_time, 0), (end_time, 0)]
        # Add start and end events
        # Add occupied_time_window events
        for occupied_start, duration, occupied_width in occupied_time_windows:
            if occupied_start < end_time and occupied_start + duration > start_time:
                events.append((max(start_time, occupied_start), occupied_width))
                events.append((min(end_time, occupied_start + duration), -occupied_width))
        # Sort by time
        events.sort()

        curr_width = bandwidth
        min_width = bandwidth
        last_event_time = start_time
        for time, width_change in events:
            if curr_width >= slice_width and time - last_event_time > slice_length:
                # # Return the first available time window <accumulate version>
                # if curr_width - width_change < slice_width or time == end_time:
                return [last_event_time, min_width, satellite_id]
                # return [last_event_time, min_width]
                # simple version

            curr_width -= width_change
            min_width = min(min_width, curr_width)
            if curr_width < slice_width:
                last_event_time = np.nan
                min_width = curr_width

            elif np.isnan(last_event_time) and curr_width >= slice_width:
                last_event_time = time
                min_width = curr_width
        return None  # Return None if there is no available time window

    def initial_graph(self):
        """
        初始化任务图
        :return: None
        """
        # initial task graph
        self.edges: List[List[Optional[Edge]]] = [[None] * self.task_num for _ in range(self.task_num)]
        for i in range(self.task_num):
            for j in range(i + 1, self.task_num):
                self.edges[i][j] = self.edges[j][i] = Edge(i, j, 1, self.initial_pheromone)

        # initial ants
        self.ants = [Ant(self.alpha, self.beta, self.task_num, self.edges) for _ in range(self.colony_size)]

    def add_pheromone(self, tour, pheromone=0.5, weight=1.0):
        """
        更新费洛蒙
        :param tour: 任务路径
        :param pheromone: 费洛蒙量
        :param weight: 更新权重
        :return: None
        """
        pheromone_to_add = self.pheromone_deposit_weight * pheromone
        for i in range(len(tour) - 1):
            self.edges[tour[i]][tour[i + 1]].pheromone += weight * pheromone_to_add

    def fade_tune_pheromone(self):
        """
        费洛蒙挥发
        :return: None
        """
        for i in range(self.task_num):
            for j in range(i + 1, self.task_num):
                self.edges[i][j].pheromone *= (1.0 - self.rho)
                if self.edges[i][j].pheromone > 10:
                    self.edges[i][j].pheromone = 10
                elif self.edges[i][j].pheromone < 0.1:
                    self.edges[i][j].pheromone = 0.1

    def run(self):
        """
        带精英策略、最大最小的蚁群算法
        :return: None
        """
        self.initial_task_world()
        self.initial_graph()
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(self.colony_size)) as executor:
            for _ in trange(int(self.steps), desc="Iteration Epoch"):

                # 1. find tours <submit>
                futures = {executor.submit(ant.find_tour) for ant in self.ants}
                concurrent.futures.wait(futures)

                # 2. store tours
                tour_list = [future.result() for future in concurrent.futures.as_completed(futures)]

                # 3. evaluate tours <submit>
                futures = {executor.submit(self.assign_tasks, tour_sequence): int(index)
                           for index, tour_sequence in enumerate(tour_list)
                           }
                concurrent.futures.wait(futures)
                score_list = [-1 for _ in range(self.colony_size)]
                for future in concurrent.futures.as_completed(futures):
                    score_list[futures[future]] = future.result()

                # 4. get the pheromone_list
                factor = 10 ** np.floor(np.log10(np.max(score_list)) + 1)
                pheromone_list = score_list / factor

                # 5. update pheromone
                for index, pheromone in enumerate(pheromone_list):
                    self.add_pheromone(tour_list[index], pheromone)

                # 6. update data
                if np.max(score_list) > self.elite["fitness"]:
                    # print("\nCongratulation, 发现新的全局最优解")
                    self.elite["fitness"] = np.max(score_list)
                    self.elite["tour"] = tour_list[score_list.index(np.max(score_list))]

                self.benefit_curve.append(self.elite["fitness"])
                # 7. elitist strategy
                self.add_pheromone(self.elite["tour"], self.elite["fitness"] / factor)
                # 8. fade pheromone
                self.fade_tune_pheromone()
        end = time.time()
        self.time = end - start
        print("执行时间:", end - start)
        # 9. parse the elite: time_list, satellite_list
        self.assign_tasks(self.elite["tour"])

    def plot_schedule(self):
        """
        绘制资源占用图
        :return:
        """
        # plot agent schedules
        fig_schedule = plt.figure(1)
        fig_schedule.suptitle("Spectrum Resource Usage", fontsize=6, x=0.5, y=0.95, ha='center', va='top')
        fig_schedule.text(0.06, 0.5, 'Bandwidth', va='center', rotation='vertical', fontsize=6)
        for satellite in range(self.satellite_num):
            ax = plt.subplot(self.satellite_num, 1, satellite + 1)
            ax.set_title("Satellite " + str(satellite), fontsize=6, pad=3)
            ax.tick_params(axis='both', which='major', labelsize=6)
            if satellite == (self.satellite_num - 1):
                ax.set_xlabel("Time", fontsize=6)
            ax.set_xlim([0, 0.5])  # 时间范围
            ax.set_ylim([0, 22])

            # 设置颜色
            # colors = ['red', 'green', 'blue', 'orange', 'yellow', 'pink', 'purple', 'black', 'gray']
            events = self.converse_task_to_events(satellite)

            curr_width = 0
            last_event_time = 0
            for time, width_change in events:
                if time > last_event_time:
                    length = time - last_event_time
                    rect = patches.Rectangle((last_event_time, 0), time - last_event_time, curr_width,
                                             facecolor='forestgreen')
                    ax.add_patch(rect)
                    # 绘图
                last_event_time = time
                curr_width += width_change

        fig_schedule.subplots_adjust(hspace=0.5)

        # set legends
        colors = ["red", "red"]
        line_styles = ["-", "-."]
        line_width_list = [10, 2]
        labels = ["Assignment Time", "Task Time"]

        def f(line_style, color_type, line_width):
            return plt.plot([], [], linestyle=line_style, color=color_type,
                            linewidth=line_width)[0]

        handles = [f(line_styles[i], colors[i], line_width_list[i]) for i in range(len(labels))]
        fig_schedule.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)

        plt.show()

    def plot_curve(self):
        """
        绘制迭代收益曲线
        :return: None
        """
        fig_convergence, ax = plt.subplots()
        # Create the plot
        iterations = range(len(self.benefit_curve))
        ax.plot(iterations, self.benefit_curve, color="red", marker='*', markersize=4)

        # Add labels and title
        ax.set_xlabel('Iterations', fontsize=8,  labelpad=2)
        ax.set_ylabel('Fitness', fontsize=8,  labelpad=2)
        ax.set_title('Algorithm Convergence Curve', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Customize the plot
        plt.grid(False)
        # plt.ylim(0, 1)

        # Show the plot
        plt.show()

    def converse_task_to_events(self, sat_index):
        """
        卫星任务集转事件集
        :param sat_index:
        :return: events
        """
        satellite = self.satellite_list[sat_index]
        occupied_windows = [[self.time_list[satellite.satellite_id][i],
                             self.tasklist[i, 3],
                             self.tasklist[i, 2]]
                            for i in satellite.execution_list
                            ]
        events = []
        # Add start and end events
        # Add occupied_time_window events
        for occupied_start, duration, occupied_width in occupied_windows:
            events.append((occupied_start, occupied_width))
            events.append((occupied_start + duration, -occupied_width))
        # Sort by time
        events.sort()
        return events

def execute_acs():
    """
    execute function
    :return:
    """
    config_file_name = "../data/configure.json"
    _colony_size = 10
    _alpha = 3.0
    _beta = 1.0
    _rho = 0.10
    _pheromone_deposit_weight = 0.001
    _initial_pheromone = 1.0
    _steps = 100
    algorithm = ACS(config_dir=config_file_name, colony_size=_colony_size, alpha=_alpha, beta=_beta, rho=_rho,
                    pheromone_deposit_weight=_pheromone_deposit_weight, initial_pheromone=_initial_pheromone,
                    steps=_steps)
    algorithm.run()
    return algorithm.time, algorithm.benefit_curve[-1], np.sum(algorithm.tasklist[:, 4])


if __name__ == "__main__":
    Config_file_name = "../data/configure.json"
    Colony_size = 10
    Alpha = 3.0
    Beta = 1.0
    Rho = 0.10
    Pheromone_deposit_weight = 0.001
    Initial_pheromone = 1.0
    Steps = 100
    Algorithm = ACS(config_dir=Config_file_name, colony_size=Colony_size, alpha=Alpha, beta=Beta, rho=Rho,
                    pheromone_deposit_weight=Pheromone_deposit_weight, initial_pheromone=Initial_pheromone,
                    steps=Steps)
    Algorithm.run()
    # 10. plot schedule
    Algorithm.plot_schedule()
    # 11. plot iteration curve
    Algorithm.plot_curve()


