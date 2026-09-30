#!/usr/bin/env python3
from task_tool.Task_generator import Task
from task_tool.HelpLibrary import Helper
from lib.Agent import Satellite
from lib.Status import Status
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import numpy as np
import random
import math
from tqdm import tqdm
import time

plt.style.use(['science', 'ieee'])

class SA(object):
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

    # algorithm parameter
    elite: dict

    # iteration curve
    benefit_curve: list

    # status
    status: Status

    time: float

    def __init__(self, config_dir, max_iterations, temperature, temperature_alpha, freezing_temperature, gama):
        # world info
        self.config_dir = config_dir
        # algorithm parameters
        self.max_iterations = max_iterations  # 最大迭代次数
        self.temperature = temperature  # 初始温度 <当前温度>
        self.freezing_temperature = freezing_temperature  # 终止温度
        self.temperature_alpha = temperature_alpha  # 降温系数
        self.gama = gama  # 扰动百分比

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
        self.satellite_list = [Satellite(self.bandwidth, i) for i in range(self.satellite_num)]
        self.tasklist = task.tasklist
        self.conflict_list = task.conflict_set
        self.task_indices = set(range(self.task_num))

        # 初始化参数
        self.score_list = [[-1] * self.task_num for _ in range(self.satellite_num)]
        self.time_list = [[-1] * self.task_num for _ in range(self.satellite_num)]

        # 初始化状态
        self.benefit_curve = []
        self.elite = {"fitness": 0,
                      "sequence": 0
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

    def construct_initial_solution(self):
        """
        构建模拟退火算法初始解
        按照权重优先生成
        :return: initial solution sequence
        """
        return np.argsort(self.tasklist[:, 4][::-1])

    def shaking(self, current_sequence):
        """
        对当前解执行扰动生成新解
        :return: new solution sequence
        """
        shaking_size = int(self.task_num * self.gama)
        if self.task_num == 0 or shaking_size < 2:
            return None
        else:
            start_index = random.randint(0, self.task_num - shaking_size)
            current_sequence[start_index: start_index + shaking_size] = \
                current_sequence[start_index: start_index + shaking_size][::-1]

            return current_sequence

    def run(self):
        """
        模拟退火算法程序主入口
        :return: None
        """
        # 1. 初始化任务世界
        self.initial_task_world()
        # 2. 生成初始解
        start = time.time()
        current_sequence = self.construct_initial_solution()
        # 3. 评估初始解
        current_fitness = self.assign_tasks(current_sequence)
        self.elite["fitness"] = current_fitness
        self.elite["sequence"] = current_sequence
        self.benefit_curve.append(self.elite["fitness"])
        # 4. 设置进度条
        max_iter = 900
        progress_bar = tqdm(total=max_iter)
        progress_bar.set_description("Iteration epoch")
        # 5. 模拟退火
        while self.temperature > self.freezing_temperature:
            for _ in range(self.max_iterations):
                # 5.1 更新进度条
                progress_bar.update(1)
                # 5.2 扰动获得邻域解
                new_sequence = self.shaking(current_sequence)
                # 5.3 评估邻域解
                new_fitness = self.assign_tasks(new_sequence)
                # 5.4 Metropolis accept
                if new_fitness > self.elite["fitness"]:
                    # 5.4.1 global
                    self.elite["fitness"] = new_fitness
                    self.elite["sequence"] = new_sequence
                    # 5.4.2 local
                    current_fitness = new_fitness
                    current_sequence = new_sequence
                elif new_fitness > current_fitness:
                    # 5.4.3 local
                    current_fitness = new_fitness
                    current_sequence = new_sequence
                elif np.random.uniform() < math.exp((new_fitness - current_fitness) / self.temperature):
                    # 5.4.4 accept
                    current_fitness = new_fitness
                    current_sequence = new_sequence
                self.benefit_curve.append(self.elite["fitness"])
            # freezing
            self.temperature *= self.temperature_alpha
        # 6. 关闭进度条
        progress_bar.close()
        end = time.time()
        self.time = end - start
        print("执行时间:", end - start)
        # 7. parse the elite: time_list, satellite_list
        self.assign_tasks(self.elite["sequence"])

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

def execute_sa():
    """
    execute function
    :return:
    """
    config_file_name = "../data/configure.json"
    _max_iterations = 5
    _temperature = 100
    _temperature_alpha = 0.95
    _freezing_temperature = 0.01
    _gama = 0.1
    algorithm = SA(config_dir=config_file_name, max_iterations=_max_iterations, temperature=_temperature,
                   temperature_alpha=_temperature_alpha, freezing_temperature=_freezing_temperature, gama=_gama)
    algorithm.run()
    return algorithm.time, algorithm.benefit_curve[-1], np.sum(algorithm.tasklist[:, 4])


if __name__ == "__main__":
    Config_file_name = "../data/configure.json"
    Max_iterations = 5
    Temperature = 100
    Temperature_alpha = 0.95
    Freezing_temperature = 0.01
    Gama = 0.1
    Algorithm = SA(config_dir=Config_file_name, max_iterations=Max_iterations, temperature=Temperature,
                   temperature_alpha=Temperature_alpha, freezing_temperature=Freezing_temperature, gama=Gama)
    Algorithm.run()
    # 8. plot schedule
    Algorithm.plot_schedule()
    # 9. plot iteration curve
    Algorithm.plot_curve()
