import numpy as np

from random import shuffle
import operator

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

# DFS graph search
def dfsSearch(Start, size, Goal):
    step = 0
    stack = []
    searched = [Start]
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
            print("solution is found! Number of nodes expanded :", step)
            print("Space complexity is: ", lengthlist[-1])
            return parent
        if find(parent, searched):
            parent_copy = parent.copy()
            searched.append(parent_copy)
        remove_from_stack = True
        list1 = findChildren(parent, size)
        for i in range(len(list1)):
            copy1 = list1[i].copy()
            children.append(copy1)
        for child in children:
            print(child)
            if find(child, searched):
                step = step + 1
                child_copy = child.copy()
                stack.append(child_copy)
                remove_from_stack = False
                break
        if remove_from_stack:
            stack.pop()
    return searched


A = constructMatrix(4, 4, 4, 1, 4, 2, 4, 3, 4, 4)
Goal = constructGoalMatrix(4, 4, 2, 2, 3, 2, 4, 2)
import time

start_time = time.time()
dfsSearch(A, 4, Goal)
print("--- %s seconds ---" % (time.time() - start_time))
