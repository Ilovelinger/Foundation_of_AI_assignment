import numpy as np

import operator
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(1000000)

from Construction import constructMatrix
from Construction import randomlypopSearchQueue
from Construction import constructGoalMatrix


# Check if the current state is the goal state
def checkIfGoalState(x, goalState):
    Aposition = goalState.index('A')
    Bposition = goalState.index('B')
    Cposition = goalState.index('C')
    if x[Aposition] == 'A' and x[Bposition] == 'B' and x[Cposition] == 'C':
        return True
    else:
        return False


# Find neighbor state for current state
def findChildren(State, size):
    children = []
    position = State.index('M')
    x = int(position / size)
    y = position - x * size
    queue = randomlypopSearchQueue(x, y, size)
    queue = queue.astype(np.int64)
    initialState = State.copy()
    while queue.shape[0] != 0:
        newState = initialState.copy()
        grid = queue[0]
        grid = grid.astype(np.int64)  # pop out a grid
        queue = np.delete(queue, 0, 0)
        column = grid[0]
        row = grid[1]
        temp = newState[column * size + row]
        newState[column * size + row] = newState[x * size + y]
        newState[x * size + y] = temp
        newState_copy = newState.copy()
        children.append(newState_copy)
    return children


# Iterative deepening search
def IDS(Start, size, Goal):
    import itertools
    def dfsWithDepth(route, depth, Goal):
        if depth == 0:
            return
        if checkIfGoalState(route[-1], Goal):
            print("Solution is found!")
            return route
        children = []
        list1 = findChildren(route[-1], size)
        for i in range(len(list1)):
            copy1 = list1[i].copy()
            children.append(copy1)
        for child in children:
            if child not in route:
                next_route = dfsWithDepth(route + [child], depth - 1, Goal)
                if next_route:
                    return next_route

    for depth in itertools.count():
        route = dfsWithDepth([Start], depth, Goal)
        print("depth is:", depth)
        if route:
            return route


A = constructMatrix(5, 5, 4, 1, 4, 2, 4, 3, 4, 4)
Goal = constructGoalMatrix(5, 5, 2, 2, 3, 2, 4, 2)
import time


start_time = time.time()
IDS(A, 5, Goal)
print("--- %s seconds ---" % (time.time() - start_time))


# #######################################################
# A1 = constructMatrix(3, 3, 1, 1, 2, 1, 3, 1, 1, 2)
# Goal1 = constructGoalMatrix(3, 3, 3, 3, 2, 3, 1, 3)
#
# A2 = constructMatrix(3, 3, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal2 = constructGoalMatrix(3, 3, 3, 3, 2, 3, 1, 3)
#
# A3 = constructMatrix(3, 3, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal3 = constructGoalMatrix(3, 3, 1, 2, 2, 2, 3, 2)
#
# A4 = constructMatrix(3, 3, 1, 1, 2, 1, 3, 1, 1, 2)
# Goal4 = constructGoalMatrix(3, 3, 1, 2, 2, 2, 3, 2)
#
# A5 = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 1, 2)
# Goal5 = constructGoalMatrix(4, 4, 3, 3, 2, 3, 1, 3)
#
# A6 = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal6 = constructGoalMatrix(4, 4, 3, 3, 2, 3, 1, 3)
#
# A7 = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal7 = constructGoalMatrix(4, 4, 1, 2, 2, 2, 3, 2)
#
# A8 = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 1, 2)
# Goal8 = constructGoalMatrix(4, 4, 1, 2, 2, 2, 3, 2)
#
# A9 = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 4, 4)
# Goal9 = constructGoalMatrix(4, 4, 3, 3, 2, 3, 1, 3)
#
# A10 = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal10 = constructGoalMatrix(4, 4, 4, 4, 3, 4, 2, 4)
#
# A11 = constructMatrix(5, 5, 1, 1, 2, 1, 3, 1, 1, 2)
# Goal11 = constructGoalMatrix(5, 5, 3, 3, 2, 3, 1, 3)
#
# A12 = constructMatrix(5, 5, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal12 = constructGoalMatrix(5, 5, 3, 3, 2, 3, 1, 3)
#
# A13 = constructMatrix(5, 5, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal13 = constructGoalMatrix(5, 5, 1, 2, 2, 2, 3, 2)
#
# A14 = constructMatrix(5, 5, 1, 1, 2, 1, 3, 1, 1, 2)
# Goal14 = constructGoalMatrix(5, 5, 1, 2, 2, 2, 3, 2)
#
# A15 = constructMatrix(5, 5, 1, 1, 2, 1, 3, 1, 5, 5)
# Goal15 = constructGoalMatrix(5, 5, 3, 3, 2, 3, 1, 3)
#
# A16 = constructMatrix(5, 5, 1, 1, 2, 1, 3, 1, 3, 3)
# Goal16 = constructGoalMatrix(5, 5, 5, 5, 4, 5, 3, 5)
#
# ##################################################
# Timelist1 = []
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A1, 3, Goal1)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# Timelist1.append(totalTime1/5)
# print("Time list1: ", Timelist1)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A2, 3, Goal2)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# Timelist1.append(totalTime1/5)
# print("Time list1: ", Timelist1)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A3, 3, Goal3)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# Timelist1.append(totalTime1/5)
# print("Time list1: ", Timelist1)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A4, 3, Goal4)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# Timelist1.append(totalTime1/5)
# print("Time list1: ", Timelist1)
#
# ##################################
# timeList2 = []
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A5, 4, Goal5)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList2.append(totalTime1/5)
# print("Time list2: ", timeList2)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A6, 4, Goal6)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList2.append(totalTime1/5)
# print("Time list2: ", timeList2)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A7, 4, Goal7)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList2.append(totalTime1/5)
# print("Time list2: ", timeList2)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A8, 4, Goal8)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList2.append(totalTime1/5)
# print("Time list2: ", timeList2)
#
# ########################################################
# timeList3 = []
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A11,5, Goal11)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList3.append(totalTime1/5)
# print("Time list3: ", timeList3)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A12,5, Goal12)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList3.append(totalTime1/5)
# print("Time list3: ", timeList3)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A13,5, Goal13)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList3.append(totalTime1/5)
# print("Time list3: ", timeList3)
#
# totalTime1 = 0
# for j in range(5):
#     start_time = time.time()
#     IDS(A14,5, Goal14)
#     totalTime1 = totalTime1 + (time.time() - start_time)
# timeList3.append(totalTime1/5)
# print("Time list3: ", timeList3)
#
# ###############################################
#
#
# x1 = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# x2 = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# x3 = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x1, Timelist1, c='r')
# plt.title("3*3 grid World")
# plt.show()
#
# plt.title("4*4 grid World")
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x2, timeList2, c='b')
# plt.show()
#
# plt.title("5*5 grid World")
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x3, timeList3, c='g')
# plt.show()