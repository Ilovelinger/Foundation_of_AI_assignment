import numpy as np
from random import shuffle


# Make a list of zeros
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


# Construct the grid world
def constructMatrix(Column, Row, Ax, Ay, Bx, By, Cx, Cy, Mx, My):
    list = zerolistmaker(Column * Row)
    Aposition = (Ax - 1) * Column + Ay - 1
    Bposition = (Bx - 1) * Column + By - 1
    Cposition = (Cx - 1) * Column + Cy - 1
    Mposition = (Mx - 1) * Column + My - 1
    list[Aposition] = 'A'
    list[Bposition] = 'B'
    list[Cposition] = 'C'
    list[Mposition] = 'M'

    return list


# Construct the goal grid world
def constructGoalMatrix(Column, Row, Ax, Ay, Bx, By, Cx, Cy):
    list = zerolistmaker(Column * Row)
    Aposition = (Ax - 1) * Column + Ay - 1
    Bposition = (Bx - 1) * Column + By - 1
    Cposition = (Cx - 1) * Column + Cy - 1
    list[Aposition] = 'A'
    list[Bposition] = 'B'
    list[Cposition] = 'C'

    return list


# Add the neighbor states into a list
def popSearchQueue(x, y, size):
    if x == 0 and y == 0:
        search_queue = np.zeros((2, 2))
        search_queue[1] += (x + 1, y)
        search_queue[0] += (x, y + 1)
    if x == 0 and y == size - 1:
        search_queue = np.zeros((2, 2))
        search_queue[0] += (x + 1, y)
        search_queue[1] += (x, y - 1)
    if x == size - 1 and y == 0:
        search_queue = np.zeros((2, 2))
        search_queue[0] += (x - 1, y)
        search_queue[1] += (x, y + 1)
    if x == size - 1 and y == size - 1:
        search_queue = np.zeros((2, 2))
        search_queue[0] += (x, y - 1)
        search_queue[1] += (x - 1, y)
    if size - 1 > x > 0 == y:
        search_queue = np.zeros((3, 2))
        search_queue[0] += (x - 1, y)
        search_queue[1] += (x, y + 1)
        search_queue[2] += (x + 1, y)
    if x == 0 and 0 < y < size - 1:
        search_queue = np.zeros((3, 2))
        search_queue[0] += (x, y - 1)
        search_queue[1] += (x + 1, y)
        search_queue[2] += (x, y + 1)
    if x == size - 1 and 0 < y < size - 1:
        search_queue = np.zeros((3, 2))
        search_queue[0] += (x, y - 1)
        search_queue[1] += (x - 1, y)
        search_queue[2] += (x, y + 1)
    if 0 < x < size - 1 == y:
        search_queue = np.zeros((3, 2))
        search_queue[0] += (x, y - 1)
        search_queue[1] += (x + 1, y)
        search_queue[2] += (x - 1, y)
    if 0 < x < size - 1 and 0 < y < size - 1:
        search_queue = np.zeros((4, 2))
        search_queue[0] += (x, y - 1)
        search_queue[1] += (x, y + 1)
        search_queue[2] += (x - 1, y)
        search_queue[3] += (x + 1, y)

    return search_queue


# Randomly add the neighbor states into a list
def randomlypopSearchQueue(x, y, size):
    if x == 0 and y == 0:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x + 1, y)
        search_queue[index2] += (x, y + 1)
    if x == 0 and y == size - 1:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x + 1, y)
        search_queue[index2] += (x, y - 1)
    if x == size - 1 and y == 0:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x - 1, y)
        search_queue[index2] += (x, y + 1)
    if x == size - 1 and y == size - 1:
        search_queue = np.zeros((2, 2))
        a = [0, 1]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x - 1, y)
    if size - 1 > x > 0 == y:
        search_queue = np.zeros((3, 2))
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue[index1] += (x - 1, y)
        search_queue[index2] += (x, y + 1)
        search_queue[index3] += (x + 1, y)
    if x == 0 and 0 < y < size - 1:
        search_queue = np.zeros((3, 2))
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x + 1, y)
        search_queue[index3] += (x, y + 1)
    if x == size - 1 and 0 < y < size - 1:
        search_queue = np.zeros((3, 2))
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x - 1, y)
        search_queue[index3] += (x, y + 1)
    if 0 < x < size - 1 == y:
        a = [0, 1, 2]
        shuffle(a)
        index1 = a[0]
        index2 = a[1]
        index3 = a[2]
        search_queue = np.zeros((3, 2))
        search_queue[index1] += (x, y - 1)
        search_queue[index2] += (x + 1, y)
        search_queue[index3] += (x - 1, y)
    if 0 < x < size - 1 and 0 < y < size - 1:
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
