# author: way
# time: 2024/6/12

import random
import matplotlib.pyplot as plt

# 优化目标1
def func1(x):
    return - (x - 1) ** 2

# 优化目标2
def func2(x):
    return - (x + 1) ** 2

# 全局种群属性
epoch = 100 # 迭代次数
pop_size = 100 # 种群数量
min_x = -100 #搜索的最小值
max_x = 100 #搜索的最大值
mutation_rate = 0.1 # 变异率

# 初始化种群
pop = [random.uniform(min_x, max_x) for _ in range(pop_size)]
func_1_epoch = [0 for _ in range(epoch)] # 优化目标1的历史记录
func_2_epoch = [0 for _ in range(epoch)] # 优化目标2的历史记录

# 交叉
def crossover(p1, p2):
    return (p1 + p2) / 2

# 变异
def mutation(p):
    mutation_rate_now = random.uniform(0, 1)
    if mutation_rate_now > mutation_rate:
        return p
    else:
        return random.uniform(min_x, max_x)


# 快速非支配排序
def non_dominated_sort(fit_func_1, fit_func_2):
    pop_size = len(fit_func_1)
    dominated_solutions = [[] for _ in range(pop_size)]
    fronts = [[]]

    domination_count = [0] * pop_size
    rank = [0] * pop_size

    # Compute domination status for each solution
    for p in range(pop_size):
        for q in range(pop_size):
            if p != q:
                if (fit_func_1[p] > fit_func_1[q] and fit_func_2[p] >= fit_func_2[q]) or \
                   (fit_func_1[p] >= fit_func_1[q] and fit_func_2[p] > fit_func_2[q]):

                    dominated_solutions[p].append(q)
                elif (fit_func_1[q] > fit_func_1[p] and fit_func_2[q] >= fit_func_2[p]) or \
                     (fit_func_1[q] >= fit_func_1[p] and fit_func_2[q] > fit_func_2[p]):
                    domination_count[p] += 1

        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    # Generate subsequent fronts
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove the last empty front
    if not fronts[-1]:
        fronts.pop()

    return fronts


# 计算拥挤度
def calculate_crowding_distance(fit_func_1, fit_func_2, front):
    length = len(front)
    crowding_distance = [0 for _ in range(len(fit_func_1))]

    # 按照fit_func_1排序
    sorted_front_1 = sorted(front, key=lambda x: fit_func_1[x])
    crowding_distance[sorted_front_1[0]] = float('inf')
    crowding_distance[sorted_front_1[-1]] = float('inf')
    for k in range(2, length - 1):
        crowding_distance[sorted_front_1[k]] += (fit_func_1[sorted_front_1[k + 1]] - fit_func_1[sorted_front_1[k - 1]]) / (fit_func_1[sorted_front_1[-1]] - fit_func_1[sorted_front_1[0]])

    # 按照fit_func_2排序
    sorted_front_2 = sorted(front, key=lambda x: fit_func_2[x])
    crowding_distance[sorted_front_2[0]] = float('inf')
    crowding_distance[sorted_front_2[-1]] = float('inf')
    for k in range(2, length - 1):
        crowding_distance[sorted_front_2[k]] += (fit_func_2[sorted_front_2[k + 1]] - fit_func_2[sorted_front_2[k - 1]]) / (fit_func_2[sorted_front_2[-1]] - fit_func_2[sorted_front_2[0]])

    return crowding_distance


# 产生新种群
def generate_new_population(population):
    # 对父代的继承
    new_population = []

    while(len(new_population) < pop_size):
        # 选择父代
        parent_1 = random.choice(population)
        parent_2 = random.choice(population)

        # 交叉
        child = crossover(parent_1, parent_2)

        # 变异
        child = mutation(child)

        new_population.append(child)

    return new_population + population

# 循环迭代
for itr in range(epoch):
    print('epoch{}'.format(itr))

    # 产生新种群
    pop = generate_new_population(pop)

    # 适应度计算
    fit_function_1 = [func1(pop[i]) for i in range(pop_size * 2)]
    fit_function_2 = [func2(pop[i]) for i in range(pop_size * 2)]

    # 快速非支配排序
    non_dominate_population = non_dominated_sort(fit_function_1, fit_function_2)

    # 拥挤度计算
    crowding_distance = []
    for i in range(0, len(non_dominate_population)):
        crowding_distance.append(calculate_crowding_distance(fit_function_1, fit_function_2, non_dominate_population[i][:]))

    offspring = []
    # 产生子代
    for i in range(0, len(non_dominate_population)):

        if len(offspring) + len(non_dominate_population[i]) <= pop_size:
            offspring += [pop[j] for j in non_dominate_population[i]]

        else:
            temp = sorted(non_dominate_population[i][:], key=lambda x: crowding_distance[i][x], reverse=True)
            for j in range(0, pop_size - len(offspring)):
                offspring.append(pop[temp[j]])

        if len(offspring) == pop_size:
            break

    pop = offspring

    # 记录历史
    func_1_epoch[itr] = max(fit_function_1)
    func_2_epoch[itr] = max(fit_function_2)

fit_function_1 = [func1(pop[i]) for i in range(pop_size)]
fit_function_2 = [func2(pop[i]) for i in range(pop_size)]


function1 = [i * -1 for i in fit_function_1]
function2 = [j * -1 for j in fit_function_2]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()

plt.plot(func_1_epoch, label='Function 1')
plt.plot(func_2_epoch, label='Function 2')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Fitness', fontsize=15)
plt.legend()
plt.show()
