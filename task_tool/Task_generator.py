#!/usr/bin/env python3
import numpy as np
from task_tool.InitWrapper import initialize_attributes


@initialize_attributes
class Task(object):
    def __init__(self, task_number, start, end, mean, std, mean2, std2):
        self.task_number = task_number
        self.tasklist = None
        self.conflict_set = None
        self.start = start
        self.end = end
        self.mean = mean
        self.std = std
        self.mean2 = mean2
        self.std2 = std2

    def task_generator(self):
        """
        按照一定分布生成任务
        :return: None
        """
        np.random.seed(666)
        band_choice = np.array(np.arange(3, 6))
        self.tasklist = np.zeros((self.task_number, 5))
        for i in range(self.task_number):
            while True:
                a = np.random.uniform(0, 1)*(self.end - self.start)
                b = a + np.random.normal(self.mean, self.std, 1)[0]
                if b <= self.end:
                    self.tasklist[i, 0] = a  # 任务开始时间
                    self.tasklist[i, 1] = b  # 任务结束时间
                    break
            self.tasklist[i, 2] = np.random.choice(band_choice, size=1)  # 任务占用频带

            while True:
                task_len = np.random.normal(self.mean2, self.std2, 1)[0]  # 任务时长
                if self.tasklist[i, 1] - self.tasklist[i, 0] > task_len:
                    self.tasklist[i, 3] = task_len
                    break

            # self.tasklist[i, 4] = np.random.randint(1, self.sc + 1)  # 任务与卫星绑定
            self.tasklist[i, 4] = np.random.uniform(0, 1) * 10  # 任务权重

        # add efficiency indicator
        efficiency = self.tasklist[:, 4] / (self.tasklist[:, 2] * self.tasklist[:, 3])
        self.tasklist = np.column_stack((self.tasklist, efficiency))

    def task_conflict_calculate(self):
        """
        构建每个任务的冲突集
        :return: None
        """
        self.conflict_set = [[] for _ in range(self.task_number)]
        for index, row in enumerate(self.tasklist):
            for sub_index, sub_row in enumerate(self.tasklist[index + 1:], start=index + 1):
                start_time1 = row[0]
                end_time1 = row[1]
                start_time2 = sub_row[0]
                end_time2 = sub_row[1]
                if not(start_time1 > end_time2 or start_time2 > end_time1):
                    self.conflict_set[index].append(sub_index)
                    self.conflict_set[sub_index].append(index)
        # converse to set the inner element
        self.conflict_set = [set(tuple(element)) for element in self.conflict_set]

    def run(self):
        self.task_generator()
        self.task_conflict_calculate()


if __name__ == "__main__":
    task = Task(600, 738188, 738188.5, 1/36, 1/288, 1/92, 1/720)
    print(task.tasklist[:, 5])
    print(task.conflict_set[1])
    # print(task.conflict_set)
