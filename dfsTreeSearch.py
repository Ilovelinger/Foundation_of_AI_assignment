import operator
import sys

import numpy as np
import matplotlib.pyplot as plt

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


# Find a target element in a list
def find(target, list):
    for i in range(len(list)):
        base = list[i]
        if operator.eq(target, base):
            return False
    return True


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


# DFS tree search
def dfsTreeSearch(Start, size, Goal):
    step = 0
    stack = []
    stack.append(Start)
    while len(stack) != 0:
        length = len(stack)
        lengthlist = []
        lengthlist.append(length)
        lengthlist.sort()
        children = []
        parent = stack[-1].copy()
        if checkIfGoalState(parent, Goal):
            step = step + 1
            print("solution is found! Number of nodes expanded(enable duplicate): ", step)
            # print("Space complexity is: ", lengthlist[-1])
            return parent, step
        remove_from_stack = True
        list1 = findChildren(parent, size)
        for i in range(len(list1)):
            copy1 = list1[i].copy()
            children.append(copy1)
        for child in children:
            # print(child)
            step = step + 1
            child_copy = child.copy()
            stack.append(child_copy)
            remove_from_stack = False
            break
        if remove_from_stack:
            stack.pop()


A = constructMatrix(4, 4, 1, 1, 2, 1, 3, 1, 1, 2)
Goal = constructGoalMatrix(4, 4, 1, 2, 2, 2, 3, 2)

import time

start_time = time.time()
dfsTreeSearch(A, 4, Goal)
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
# Steplist1 = []
# totalStep1 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A1, 3, Goal1)
#     totalStep1 = totalStep1 + temp
# Steplist1.append(totalStep1 / 7)
#
# totalStep2 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A2, 3, Goal2)
#     totalStep2 = totalStep2 + temp
# Steplist1.append(totalStep2 / 7)
#
# totalStep3 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A3, 3, Goal3)
#     totalStep3 = totalStep3 + temp
# Steplist1.append(totalStep3 / 7)
#
# totalStep4 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A4, 3, Goal4)
#     totalStep4 = totalStep4 + temp
# Steplist1.append(totalStep4 / 7)
#
# ##################################
# Steplist2 = []
#
# totalStep5 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A5, 4, Goal5)
#     totalStep5 = totalStep5 + temp
# Steplist2.append(totalStep5 / 7)
#
# totalStep6 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A6, 4, Goal6)
#     totalStep6 = totalStep6 + temp
# Steplist2.append(totalStep6 / 7)
#
# totalStep7 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A7, 4, Goal7)
#     totalStep7 = totalStep7 + temp
# Steplist2.append(totalStep7 / 7)
#
# totalStep8 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A8, 4, Goal8)
#     totalStep8 = totalStep8 + temp
# Steplist2.append(totalStep8 / 7)
#
# totalStep9 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A9, 4, Goal9)
#     totalStep9 = totalStep9 + temp
# Steplist2.append(totalStep9 / 7)
#
# totalStep10 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A10, 4, Goal10)
#     totalStep10 = totalStep10 + temp
# Steplist2.append(totalStep10 / 7)
#
# ########################################################
# Steplist3 = []
#
# totalStep12 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A11, 5, Goal11)
#     totalStep12 = totalStep12 + temp
# Steplist3.append(totalStep12 / 7)
#
# totalStep12 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A12, 5, Goal12)
#     totalStep12 = totalStep12 + temp
# Steplist3.append(totalStep12 / 7)
#
# totalStep14 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A13, 5, Goal13)
#     totalStep14 = totalStep14 + temp
# Steplist3.append(totalStep14 / 7)
#
# totalStep14 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A14, 5, Goal14)
#     totalStep14 = totalStep14 + temp
# Steplist3.append(totalStep14 / 7)
#
# totalStep15 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A15, 5, Goal15)
#     totalStep15 = totalStep15 + temp
# Steplist3.append(totalStep15 / 7)
#
# totalStep16 = 0
# for j in range(7):
#     parent, temp = dfsTreeSearch(A16, 5, Goal16)
#     totalStep16 = totalStep16 + temp
# Steplist3.append(totalStep16 / 7)
