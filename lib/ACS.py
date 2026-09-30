import random


class Edge:
    a: int  # start
    b: int  # end
    weight: int  # weight-distance
    pheromone: float

    def __init__(self, a, b, weight, initial_pheromone):
        self.a = a
        self.b = b
        self.weight = weight
        self.pheromone = initial_pheromone


class Ant:
    alpha: float  # 费洛蒙重要因子
    beta: float  # 启发式信息重要因子
    num_nodes: int  # 节点数目<任务+1>
    edges: list[list[Edge]]  # 任务图
    tour: list[int]  # 任务链

    def __init__(self, alpha, beta, num_nodes, edges):
        self.alpha = alpha
        self.beta = beta
        self.num_nodes = num_nodes
        self.edges = edges

    def _select_node(self):
        """
        轮盘赌选择下一个节点
        :return: unvisited_node：：下一个任务节点
        """
        roulette_wheel = 0.0
        unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
        heuristic_total = 0.0
        for unvisited_node in unvisited_nodes:
            heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
        for unvisited_node in unvisited_nodes:
            roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                              pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
        random_value = random.uniform(0.0, roulette_wheel)
        wheel_position = 0.0
        for unvisited_node in unvisited_nodes:
            wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                              pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            if wheel_position >= random_value:
                return unvisited_node

    def find_tour(self):
        """
        蚂蚁搜索路径
        :return: 蚂蚁一条遍历路径
        """
        # 1. random generate a jump-off point
        self.tour = [random.randint(0, self.num_nodes - 1)]

        # 2. find the whole tour node by node
        while len(self.tour) < self.num_nodes:
            self.tour.append(self._select_node())

        return self.tour

