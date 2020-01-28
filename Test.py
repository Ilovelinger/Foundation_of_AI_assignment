import matplotlib.pyplot as plt

# def CreateRoad():
#     road = []
#     road = input("please enter the road: ")
#     road.split()
#     if len(road) > 100:
#         print("The length of the road is lager than 100! Please reenter")
#         return CreateRoad()
#     else:
#         return road
#
#
# def checkifSussess(road):
#     attack = 100
#     immune = False
#     for i in range(len(road)):
#         if road[i] == 'G':
#             attack = attack * 2
#             if attack >= 10000:
#                 attack = 10000
#         if road[i] == 'S':
#             immune = True
#         if road[i] == 'B':
#             if attack >= 1000 or immune == True:
#                 print('Yes')
#                 exit()
#             else:
#                 print("No")
#                 exit()
#
#
# # checkifSussess(CreateRoad())
#
# x = range(1, 7)
# error = [14440.775919215097,
#          5161.670475775598,
#          2730.822528151537,
#          1968.0657614613988,
#          1579.4715610185244,
#          1287.0370306251166
#          ]
# plt.plot(x, error)
# plt.xlabel('k')
# plt.ylabel('distortion  measure')
# plt.scatter(3, error[2], s=50, color='k')
# plt.show()


Timelist2 = [231.1053876876831, 459.31762142181395, 5.713824939727783, 3.0217583656311033]

Timelist1 = [11.235149478912353, 18.46364860534668, 0.5763772487640381, 0.48241539001464845]

Timelist3 = [804.4988111019135, 1844.5922655582428, 12.87375054359436, 7.3520880222320555]

# list1 = [None,None,None, 1618213]
# list2 = []


x = ["D1=6, D2=10", "D1=9, D2=10", "D1=9, D2=3", "D1=6, D2=3"]
plt.xlabel("Problem Difficulty")
plt.ylabel("Computational time")
plt.plot(x, Timelist1, c='r')
plt.title("3*3 grid World")
plt.show()

plt.title("4*4 grid World")
plt.xlabel("Problem Difficulty")
plt.ylabel("Computational time")
plt.plot(x, Timelist2, c='b')
plt.show()

plt.title("5*5 grid World")
plt.xlabel("Problem Difficulty")
plt.ylabel("Computational time")
plt.plot(x, Timelist3, c='g')
plt.show()
