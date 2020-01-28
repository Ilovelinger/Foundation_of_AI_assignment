import numpy as np

import operator
import matplotlib.pyplot as plt
from random import shuffle

from Construction import constructMatrix
from Construction import randomlypopSearchQueue
from Construction import constructGoalMatrix


# Define the node class
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, state=None):
        self.parent = parent
        self.state = state

        self.g = 0
        self.h = 0
        self.f = 0


# Check is the current state is the goal state
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


# Compute the Manhattan distance
def Manhattan(state, goalState, size):
    Aposition = state.index('A')
    Ax = int(Aposition / size)
    Ay = Aposition - Ax * size

    Bposition = state.index('B')
    Bx = int(Aposition / size)
    By = Bposition - Bx * size

    Cposition = state.index('C')
    Cx = int(Cposition / size)
    Cy = Cposition - Cx * size

    goalAposition = goalState.index('A')
    goalAx = int(goalAposition / size)
    goalAy = goalAposition - goalAx * size

    goalBposition = goalState.index('B')
    goalBx = int(goalBposition / size)
    goalBy = goalBposition - goalBx * size

    goalCposition = goalState.index('C')
    goalCx = int(goalCposition / size)
    goalCy = goalCposition - goalCx * size

    manhattenDistance = 10 * (
            abs(goalAx - Ax) + abs(goalAy - Ay) + abs(goalBx - Bx) + abs(goalBy - By) + abs(goalCx - Cx) + abs(
        goalCy - Cy))

    return manhattenDistance


# Find the reachable state from the current state
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
        new_node = Node(initialState, newState)
        children.append(new_node)
    return children


# Compute the move cost
def computeMoveCost(fathermovecost):
    movecost = fathermovecost + 10
    return movecost


# Find minimum evaluation F
def findminF(movecost, list, Goal):
    Flist = []
    for i in range(len(list)):
        F = movecost + Manhattan(list[i], Goal)
        Flist.append(F)
    Flist.sort()
    Findex = Flist.index(max(Flist))
    currentState = list.pop(Findex)
    return currentState


# Find a node in open list
def findInopenList(a, b):
    for open_node in b:
        if a.state == open_node.state and a.g > open_node.g:
            return True


# Find a node in close list
def findIncloseList(a, b):
    for closed_node in b:
        if a.state == closed_node.state:
            return True


# Find node
def findNode(state, list):
    for node in list:
        if node.state == state:
            return node


# A star search algorithm
def AstarSearch(Start, size, Goal):
    step = 0
    openlist = []
    closelist = []
    start_node = Node(None, Start)
    start_node.g = start_node.h = start_node.f = 0
    openlist.append(start_node)
    closelist.append(start_node)
    while len(openlist) != 0:
        current_node = openlist[0]
        current_index = 0
        for index, item in enumerate(openlist):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        step = step + 1
        if checkIfGoalState(current_node.state, Goal):
            route = [current_node.state]
            end = True
            current = current_node
            while end:
                parent = current.parent
                route = route + [parent]
                current = findNode(parent, closelist)
                if parent == Start:
                    end = False
            route.reverse()
            print("The path is: ", route)
            print("solution is found!")
            print("Number of nodes expanded: ", step)
            print(current_node.state)
            return step
        print(current_node.state)
        # print("Current step cost is: ", step)
        openlist.pop(current_index)
        closelist.append(current_node)
        children = findChildren(current_node.state, size)
        for child in children:
            # Child is on the closed list
            whetherincloselist = findIncloseList(child, closelist)
            if whetherincloselist:
                continue
            # Create the f, g, and h values
            child.g = current_node.g + 10
            child.h = Manhattan(child.state, Goal, size)
            child.f = child.g + child.h
            whetherinopenlist = findIncloseList(child, openlist)
            # Child is already in the open list
            if whetherinopenlist:
                continue
            openlist.append(child)


A = constructMatrix(4, 4, 4, 1, 4, 2, 4, 3, 4, 4)
Goal = constructGoalMatrix(4, 4, 2, 2, 3, 2, 4, 2)
import time

start_time = time.time()
print(AstarSearch(A, 4, Goal))
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
#     temp = AstarSearch(A1, 3, Goal1)
#     totalStep1 = totalStep1 + temp
# Steplist1.append(totalStep1 / 7)
#
# totalStep2 = 0
# for j in range(7):
#     temp = AstarSearch(A2, 3, Goal2)
#     totalStep2 = totalStep2 + temp
# Steplist1.append(totalStep2 / 7)
#
# totalStep3 = 0
# for j in range(7):
#     temp = AstarSearch(A3, 3, Goal3)
#     totalStep3 = totalStep3 + temp
# Steplist1.append(totalStep3 / 7)
#
# totalStep4 = 0
# for j in range(7):
#     temp = AstarSearch(A4, 3, Goal4)
#     totalStep4 = totalStep4 + temp
# Steplist1.append(totalStep4 / 7)
#
# ##################################
# Steplist2 = []
#
# totalStep5 = 0
# for j in range(7):
#     temp = AstarSearch(A5, 4, Goal5)
#     totalStep5 = totalStep5 + temp
# Steplist2.append(totalStep5 / 7)
#
# totalStep6 = 0
# for j in range(7):
#     temp = AstarSearch(A6, 4, Goal6)
#     totalStep6 = totalStep6 + temp
# Steplist2.append(totalStep6 / 7)
#
# totalStep7 = 0
# for j in range(7):
#     temp = AstarSearch(A7, 4, Goal7)
#     totalStep7 = totalStep7 + temp
# Steplist2.append(totalStep7 / 7)
#
# totalStep8 = 0
# for j in range(7):
#     temp = AstarSearch(A8, 4, Goal8)
#     totalStep8 = totalStep8 + temp
# Steplist2.append(totalStep8 / 7)
#
# ########################################################
# Steplist3 = []
#
# totalStep12 = 0
# for j in range(7):
#     temp = AstarSearch(A11, 5, Goal11)
#     totalStep12 = totalStep12 + temp
# Steplist3.append(totalStep12 / 7)
#
# totalStep12 = 0
# for j in range(7):
#     temp = AstarSearch(A12, 5, Goal12)
#     totalStep12 = totalStep12 + temp
# Steplist3.append(totalStep12 / 7)
#
# totalStep14 = 0
# for j in range(7):
#     temp = AstarSearch(A13, 5, Goal13)
#     totalStep14 = totalStep14 + temp
# Steplist3.append(totalStep14 / 7)
#
# totalStep14 = 0
# for j in range(7):
#     temp = AstarSearch(A14, 5, Goal14)
#     totalStep14 = totalStep14 + temp
# Steplist3.append(totalStep14 / 7)
#
# ###############################################
#
# print("Step list1: ", Steplist1)
# print("Step list2: ", Steplist2)
# print("Step list3: ", Steplist3)
#
# x1 = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# x2 = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# x3 = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x1, Steplist1, c='r')
# plt.title("3*3 grid World")
# plt.show()
#
# plt.title("4*4 grid World")
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x2, Steplist2, c='b')
# plt.show()
#
# plt.title("5*5 grid World")
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x3, Steplist3, c='g')
# plt.show()
