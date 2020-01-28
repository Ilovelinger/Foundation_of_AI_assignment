from collections import deque

import numpy
import numpy as np
import matplotlib.pyplot as plt
import time

import operator

from Construction import constructMatrix
from Construction import popSearchQueue
from Construction import constructGoalMatrix


# Define the class node
class Node():

    def __init__(self, parent=None, state=None):
        self.parent = parent
        self.state = state


# Check for goal state
def checkIfGoalState(x, goalState):
    Aposition = goalState.index('A')
    Bposition = goalState.index('B')
    Cposition = goalState.index('C')
    if x[Aposition] == 'A' and x[Bposition] == 'B' and x[Cposition] == 'C':
        return True
    else:
        return False


# Find target element in a list
def find(target, list):
    for i in range(len(list)):
        base = list[i]
        if operator.eq(target, base):
            return False
    return True


# Find node in a list
def findNode(state, list):
    for node in list:
        if node.state == state:
            return node
    print("The node is not found!")


# Methods for moving agent to its neighbor grid
def moveAgent(State, size, Goal, searchedNode, Start, parentOfStartState):
    endloop = False
    step = 0
    position = State.index('M')
    x = int(position / size)
    y = position - x * size
    queue = popSearchQueue(x, y, size)
    queue = queue.astype(np.int64)
    parentState = []
    initialState = State.copy()
    initialNode = Node(parentOfStartState, initialState)
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
        # print(newState)
        if checkIfGoalState(new_node.state, Goal):
            searchedNode.append(initialNode)
            route = [new_node.state]
            end = True
            current = new_node
            while end:
                parent = current.parent
                route = route + [parent]
                current = findNode(parent, searchedNode)
                if parent == Start:
                    end = False
            route.reverse()
            print("Solution is found!")
            print("The shortest path is: ", route)
            endloop = True
            break
        else:
            parentState.append(new_node)
            step = step + 1
    return parentState, initialState, step * 2 + 1, endloop


# BFS graph search
def BFSsearch(Start, size, Goal):
    start_node = Node(None, Start)
    searchedNodes = []
    searchedState = []
    searchedNodes.append(start_node)
    searchedState.append(Start)
    parentState, searchedInitialState, totalstep, end = moveAgent(Start, size, Goal, searchedNodes, Start, None)
    while len(parentState) != 0:
        length = len(parentState)
        lengthlist = []
        lengthlist.append(length)
        lengthlist.sort()
        State = parentState.pop(0)
        if not find(State.state, searchedState):
            continue
        statelist1, b, step, end = moveAgent(State.state, size, Goal, searchedNodes, Start, State.parent)
        totalstep = totalstep + step
        # print("Number of nodes expanded: ", totalstep)
        if end:
            break
        for i in range(len(statelist1)):
            parentState.append(statelist1[i])
        searchedNodes.append(State)
        searchedState.append(State.state)
    return totalstep
    # print("Space comlexity is : ", lengthlist[-1])


A = constructMatrix(4, 4, 4, 1, 4, 2, 4, 3, 4, 4)
Goal = constructGoalMatrix(4, 4, 2, 2, 3, 2, 4, 2)
import time

start_time = time.time()
BFSsearch(A, 4, Goal)
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
# temp = BFSsearch(A1, 3, Goal1)
# totalStep1 = totalStep1 + temp
# Steplist1.append(totalStep1)
#
# totalStep2 = 0
#
# temp = BFSsearch(A2, 3, Goal2)
# totalStep2 = totalStep2 + temp
# Steplist1.append(totalStep2)
#
# totalStep3 = 0
#
# temp = BFSsearch(A3, 3, Goal3)
# totalStep3 = totalStep3 + temp
# Steplist1.append(totalStep3)
#
# totalStep4 = 0
# temp = BFSsearch(A4, 3, Goal4)
# totalStep4 = totalStep4 + temp
# Steplist1.append(totalStep4)
#
# ##################################
# Steplist2 = []
#
# totalStep5 = 0
#
# temp = BFSsearch(A5, 4, Goal5)
# totalStep5 = totalStep5 + temp
# Steplist2.append(totalStep5)
#
# totalStep6 = 0
#
# temp = BFSsearch(A6, 4, Goal6)
# totalStep6 = totalStep6 + temp
# Steplist2.append(totalStep6)
#
# totalStep7 = 0
#
# temp = BFSsearch(A7, 4, Goal7)
# totalStep7 = totalStep7 + temp
# Steplist2.append(totalStep7)
#
# totalStep8 = 0
#
# temp = BFSsearch(A8, 4, Goal8)
# totalStep8 = totalStep8 + temp
# Steplist2.append(totalStep8)
#
# ########################################################
# Steplist3 = []
#
# totalStep12 = 0
#
# temp = BFSsearch(A11, 5, Goal11)
# totalStep12 = totalStep12 + temp
# Steplist3.append(totalStep12)
#
# totalStep12 = 0
#
# temp = BFSsearch(A12, 5, Goal12)
# totalStep12 = totalStep12 + temp
# Steplist3.append(totalStep12)
#
# totalStep14 = 0
#
# temp = BFSsearch(A13, 5, Goal13)
# totalStep14 = totalStep14 + temp
# Steplist3.append(totalStep14)
#
# totalStep14 = 0
#
# temp = BFSsearch(A14, 5, Goal14)
# totalStep14 = totalStep14 + temp
# Steplist3.append(totalStep14)
#
#
#
# x = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x, Steplist1, c='r')
# plt.title("3*3 grid World")
# plt.show()
#
# plt.title("4*4 grid World")
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x, Steplist2, c='b')
# plt.show()
#
# plt.title("5*5 grid World")
# plt.xlabel("Problem Difficulty")
# plt.ylabel("Number of nodes expanded")
# plt.plot(x, Steplist3, c='g')
# plt.show()
