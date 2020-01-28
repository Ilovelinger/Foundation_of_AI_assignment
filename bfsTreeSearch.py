import numpy as np

import operator
import matplotlib.pyplot as plt
from Construction import constructMatrix
from Construction import popSearchQueue
from Construction import constructGoalMatrix


# Define the node class
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


# Find a target element in a list
def find(target, list):
    for i in range(len(list)):
        base = list[i]
        if operator.eq(target, base):
            return False
    return True

# Find a node in a list
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
            print("The shortest path is: ", route)
            print("Solution is found!")
            endloop = True
            break
        else:
            parentState.append(new_node)
            step = step + 1
    return parentState, initialState, step, endloop

# BFS tree search
def BFSTreesearch(Start, size, goal):
    start_node = Node(None, Start)
    searchedNodes = []
    searchedNodes.append(start_node)
    parentState, searchedInitialState, totalstep, end = moveAgent(Start, size, goal, searchedNodes, Start, None)
    print("Number of nodes expanded(enable duplicate): ", totalstep)
    while len(parentState) != 0:
        length = len(parentState)
        lengthlist = []
        lengthlist.append(length)
        lengthlist.sort()
        State = parentState.pop(0)
        statelist1, b, step, end = moveAgent(State.state, size, goal, searchedNodes, Start, State.parent)
        totalstep = totalstep + step
        print("Number of nodes expanded(enable duplicate): ", totalstep)
        if end:
            break
        for i in range(len(statelist1)):
            parentState.append(statelist1[i])
        searchedNodes.append(State)
        # b_copy = b.copy()
    return totalstep
    print("Space comlexity is : ", lengthlist[-1])


# A = constructMatrix(4, 4, 4, 1, 4, 2, 4, 3, 4, 4)
# Goal = constructGoalMatrix(4, 4, 2, 2, 3, 2, 4, 2)

A2 = constructMatrix(3, 3, 1, 1, 2, 1, 3, 1, 3, 3)
Goal2 = constructGoalMatrix(3, 3, 3, 3, 2, 3, 1, 3)

import time

start_time = time.time()
BFSTreesearch(A2, 3, Goal2)
print("--- %s seconds ---" % (time.time() - start_time))


