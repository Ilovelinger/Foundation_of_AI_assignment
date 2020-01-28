import numpy as np

import operator

from random import shuffle

import sys

sys.setrecursionlimit(1000000)

A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'A', 'B', 'C', 'M']


def checkIfGoalState(x):
    if x[5] == 'A' and x[9] == 'B' and x[13] == 'C':
        return True
    else:
        return False


def popSearchQueue(x, y):
    if x == 0 and y == 0:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x + 1, y)
        search_queue[index2] += (x, y + 1)
    if x == 0 and y == 3:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x + 1, y)
        search_queue[index2] += (x, y - 1)
    if x == 3 and y == 0:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x - 1, y)
        search_queue[index2] += (x, y + 1)
    if x == 3 and y == 3:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x - 1, y)
    if 3 > x > 0 == y:
        search_queue = np.zeros((3, 2))
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue[index1] += (x - 1, y)
        search_queue[index2] += (x, y + 1)
        search_queue[index3] += (x + 1, y)
    if x == 0 and 0 < y < 3:
        search_queue = np.zeros((3, 2))
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x + 1, y)
        search_queue[index3] += (x, y + 1)
    if x == 3 and 0 < y < 3:
        search_queue = np.zeros((3, 2))
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x - 1, y)
        search_queue[index3] += (x, y + 1)
    if 0 < x < 3 == y:
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue = np.zeros((3, 2))
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x + 1, y)
        search_queue[index3] += (x - 1, y)
    if 0 < x < 3 and 0 < y < 3:
        a = [0, 1, 2, 3]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        index4 = a[3]
        search_queue = np.zeros((4, 2))
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x, y + 1)
        search_queue[index3] += (x - 1, y)
        search_queue[index4] += (x + 1, y)

    return search_queue


def find(target, list):
    for i in range(len(list)):
        base = list[i]
        if operator.eq(target, base):
            return False
    return True


def findChildren(State):
    children = []
    position = State.index('M')
    x = int(position / 4)
    y = position - x * 4
    queue = popSearchQueue(x, y)
    queue = queue.astype(np.int64)
    initialState = State.copy()
    while queue.shape[0] != 0:
        newState = initialState.copy()
        grid = queue[0]
        grid = grid.astype(np.int64)  # pop out a grid
        queue = np.delete(queue, 0, 0)
        column = grid[0]
        row = grid[1]
        temp = newState[column * 4 + row]
        newState[column * 4 + row] = newState[x * 4 + y]
        newState[x * 4 + y] = temp
        newState_copy = newState.copy()
        children.append(newState_copy)
    return children


def dfsSearchWithRecursion(route):  # havent compute the depth and steps
    depth = 0
    if checkIfGoalState(route[-1]):
        print("Solution is found!")
        print("Current depth: ", depth)
        return route
    children = []
    list1 = findChildren(route[-1])
    for i in range(len(list1)):
        copy1 = list1[i].copy()
        children.append(copy1)
    for child in children:
        print(child)
        if child not in route:
            next_route = dfsSearchWithRecursion(route + [child])
            if next_route:
                return next_route


print(dfsSearchWithRecursion([A]))
