#!/usr/bin/env python3
import time
from task_tool.Task_generator import Task
from task_tool.HelpLibrary import Helper
from lib.Agent import Satellite
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import concurrent.futures
import numpy as np
import heapq
import os
import random
from tqdm import trange

plt.style.use(['science', 'ieee'])

class GA(object):
    bandwidth: int
    task_num: int
    satellite_num: int
    tasklist: np.ndarray
    task_indices: set
    conflict_list: list
    satellite_list: list[Satellite]

    # main result
    time_list: list  # 2D list

    # algorithm parameters
    population: np.ndarray
    fitness: list
    elite: dict

    time: float

    # iteration curve
    benefit_curve: list

    def __init__(self, config_dir, population_size, max_generation, mutation_rate, tournament_size, crossover_rate):
        # world info
        self.config_dir = config_dir
        # algorithm parameter
        self.population_size = population_size
        self.max_generation = max_generation
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate

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

        # 初始化参数
        self.time_list = [[-1] * self.task_num for _ in range(self.satellite_num)]

        # 初始化状态
        self.fitness = [-1 for _ in range(self.population_size)]
        self.benefit_curve = []
        self.elite = {"fitness": 0,
                      "chromosome": 0
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

    def initial_population(self):
        """
        初始化种群
        :return: None
        """
        # 1. construct <population size> same chromosome
        self.population = np.tile(np.arange(self.task_num), (self.population_size, 1))

        # 2. ramdom shuffle all the individuals
        for individual in self.population:
            np.random.shuffle(individual)

    def arrange_task_for_score(self, population=None):
        """
        分配任务，获取得分
        :param population:
        :return: None or fitness
        """
        return_status = True
        if population is None:
            population = self.population
            return_status = False

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()//8) as executor:
            futures = {executor.submit(self.assign_tasks, individual): int(index)
                       for index, individual in enumerate(population)
                       }
            concurrent.futures.wait(futures)

            if not return_status:
                for future in concurrent.futures.as_completed(futures):
                    self.fitness[futures[future]] = future.result()
            else:
                new_born_fitness = [-1 for _ in range(len(population))]
                for future in concurrent.futures.as_completed(futures):
                    new_born_fitness[futures[future]] = future.result()
                return new_born_fitness

    def selection(self):
        """
        锦标赛选择
        :return: individual_1, individual_2
        """
        tournament_1 = random.sample(self.fitness, self.tournament_size)
        tournament_2 = random.sample(self.fitness, self.tournament_size)
        return self.fitness.index(max(tournament_1)), self.fitness.index(max(tournament_2))

    def crossover(self, parent1, parent2):
        """
        交叉算子
        :param parent1:
        :param parent2:
        :return: offspring1, offspring2
        """
        # 1. Generate the crossover points
        point1, point2 = np.sort(np.random.choice(self.task_num, 2, replace=False))
        child1, child2 = np.zeros(self.task_num, dtype=int), np.zeros(self.task_num, dtype=int)

        # 2. First, keep the genes between the crossover points as in the parents
        child1[point1:point2 + 1], child2[point1:point2 + 1] = self.population[parent1][point1:point2 + 1], \
            self.population[parent2][point1:point2 + 1]

        # 3. Then, correct the genes that cause conflict
        for i in list(range(point1)) + list(range(point2 + 1, self.task_num)):
            c1, c2 = self.population[parent2][i], self.population[parent1][i]
            while c1 in child1[point1:point2 + 1]:
                i1 = np.where(self.population[parent1] == c1)[0]
                c1 = self.population[parent2][i1]
            while c2 in child2[point1:point2 + 1]:
                i2 = np.where(self.population[parent2] == c2)[0]
                c2 = self.population[parent1][i2]

            child1[i], child2[i] = c1, c2

        return child1, child2

    @staticmethod
    def mutation(individual):
        """
        变异算子
        :param individual:
        :return: individual
        """
        # 1. Choose start and end indices for the slice to reverse
        start, end = np.sort(np.random.choice(len(individual), 2, replace=False))
        # 2. Reverse the selected slice
        individual[start:end + 1] = individual[start:end + 1][::-1]

        return individual

    def survival(self, new_born_fitness, new_born):
        """
        选出能生存
        :return: new population
        """
        total_fitness = self.fitness + new_born_fitness
        next_generation = heapq.nlargest(self.population_size, range(len(total_fitness)), key=total_fitness.__getitem__)

        next_generation_old = [int(i) for i in next_generation if i < self.population_size]
        next_generation_new = [int(i) - self.population_size for i in next_generation if i >= self.population_size]

        self.fitness = [self.fitness[i] for i in next_generation_old]+[new_born_fitness[i] for i in next_generation_new]

        self.population = self.population[next_generation_old, :]
        try:
            survived_new_born = new_born[next_generation_new, :]
            self.population = np.vstack((self.population, survived_new_born))
        except IndexError:
            pass
            # with np.printoptions(threshold=np.inf, linewidth=200, precision=3, suppress=True):
            #     print("下一代{}：".format(self.population))

    def elite_search(self):
        """
        记录全局最优解
        :return: None
        """
        # 1. process global data
        if max(self.fitness) > self.elite["fitness"]:
            # 1. get index
            elite_index = self.fitness.index(max(self.fitness))
            # 2. record
            self.elite["fitness"] = self.fitness[elite_index]
            self.elite["chromosome"] = self.population[elite_index]
        # 2. process curve
        self.benefit_curve.append(self.elite["fitness"])

    def run(self):
        """
        遗传算法主入口
        :return: None
        """
        # 1. 初始化任务世界
        self.initial_task_world()

        # 2. 生成初始解
        start = time.time()
        self.initial_population()

        # 3. 评估
        self.arrange_task_for_score()
        # 4. 记录精英
        self.elite_search()
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 8) as executor:
            for _ in trange(self.max_generation, desc="Iteration epoch"):
                new_born = []
                # 1. select
                futures = {executor.submit(self.selection) for _ in iter(range(int(1 / 2 * self.population_size)))
                           if np.random.random() < self.crossover_rate
                           }

                concurrent.futures.wait(futures)
                for future in concurrent.futures.as_completed(futures):
                    parent1, parent2 = future.result()

                    # 2.crossover
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    # 3. mutate
                    if np.random.random() < self.mutation_rate:
                        offspring1 = self.mutation(offspring1)
                    if np.random.random() < self.mutation_rate:
                        offspring2 = self.mutation(offspring2)
                    new_born += [offspring1, offspring2]

                new_born = np.array(new_born)
                # 4. evaluate the newborns
                new_born_fitness = self.arrange_task_for_score(new_born)
                # 5. fit the environment ones survival
                self.survival(new_born_fitness, new_born)
                self.elite_search()
        end = time.time()
        self.time = end - start
        print("运行时间：", end - start)
        # 6. parse the elite: time_list, satellite_list
        self.assign_tasks(self.elite["chromosome"])

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

def execute_ga():
    """
    execute function
    :return:
    """
    _config_file_name = "../data/configure.json"
    _population_size = 50
    _max_generation = 100
    _mutation_rate = 0.2
    _tournament_size = 5
    _crossover_rate = 0.3

    algorithm = GA(config_dir=_config_file_name, population_size=_population_size, max_generation=_max_generation,
                   mutation_rate=_mutation_rate, tournament_size=_tournament_size, crossover_rate=_crossover_rate)
    algorithm.run()

    return algorithm.time, algorithm.benefit_curve[-1], np.sum(algorithm.tasklist[:, 4])


if __name__ == "__main__":
    Config_file_name = "../data/configure.json"
    Population_size = 50
    Max_generation = 100
    Mutation_rate = 0.2
    Tournament_size = 5
    Crossover_rate = 0.3
    Algorithm = GA(config_dir=Config_file_name, population_size=Population_size, max_generation=Max_generation,
                   mutation_rate=Mutation_rate, tournament_size=Tournament_size, crossover_rate=Crossover_rate)
    Algorithm.run()
    # 7. plot schedule
    Algorithm.plot_schedule()
    # 8. plot iteration curve
    Algorithm.plot_curve()
