#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:59:26 2019

@author: nakul
"""

import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

show_animation = True
# =============================================================================
# plt.plot(ox,oy,".k")           
# plt.axis([0, 250, 0, 150])
# plt.grid(True)
# =============================================================================
ox,oy = [],[]
#explore_x,explore_y = [],[]
obstacle = np.zeros(shape=(1110,1010))
m = 0
res = 1
for x in range (1110):    
    for y in range(1010):
        
#Equations for circles
        c1 = (x - round(390/res))**2 + (y - round(960/res))**2 - ((40.5/res) + m)**2
        
        c2 = (x - round(438 / res)) ** 2 + (y - round(736 / res)) ** 2 - ((40.5 / res) + m) ** 2
        
        c3 = (x - round(438 / res)) ** 2 + (y - round(274 / res)) ** 2 - ((40.5 / res) + m) ** 2
        
        c4 = (x - round(390 / res)) ** 2 + (y - round(45 / res)) ** 2 - ((40.5 / res) + m) ** 2

#Equation for table
        t1 = -y + (750.1/res) - m
        t2 = y - (910/res) - m
        t3 = -x + (149.89/res)
        t4 = x - (309.79/res) 
     
        #Table left circle
        t_l = (x - (149.89/res))**2 + (y - (830.5/res))**2 - ((79.89/res) + m)**2
        
        #Table right circle
        t_r = (x - (309.79/res))**2 + (y - (830.5/res))**2 - ((79.89/res) + m)**2

# Rectangle-1
        f1 = -y + (0/res) - m
        f2 = y - (35/res) - m
        f3 = -x + (685/res) - m
        f4 = x - (1110/res) - m

#Rectangle -2
        f5 = -y + (35 / res) - m
        f6 = y - (111 / res) - m
        f7 = -x + (927 / res) - m
        f8 = x - (1110 / res) - m        

#Rectangle-3
        f9 = -y + (35 / res) - m
        f10 = y - (93 / res) - m
        f11 = -x + (779 / res) - m
        f12 = x - (896 / res) - m

#Rectangle-4
        f13 = -y + (35 / res) - m
        f14 = y - (187 / res) - m
        f15 = -x + (474 / res) - m
        f16 = x - (748 / res) - m  
    
#Rectangle-5    
        f17 = -y + (919/res) - m
        f18 = y - (1010/res) - m
        f19 = -x + (983/res) - m
        f20 = x - (1026/res) - m
          
#Rectangle-6
        f20_1 = -y + (827/res) - m
        f21 = y - (1010/res) - m
        f22 = -x + (832/res) - m
        f23 = x - (918/res) - m
        
#Rectangle-7
        f24 = -y + (621/res) - m
        f25 = y - (697/res) - m
        f26 = -x + (744/res) - m
        f27 = x - (1110/res) - m  
    
#Rectangle-8
        f28 = -y + (449/res) - m
        f29 = y - (566/res) - m
        f30 = -x + (1052/res) - m
        f31 = x - (1110/res) - m
        
#Rectangle-9
        f32 = -y + (363/res) - m
        f33 = y - (449/res) - m
        f34 = -x + (1019/res) - m
        f35 = x - (1110/res) - m
        
#Rectangle-10
        f36 = -y + (178.75/res) - m
        f37 = y - (295.75/res) - m
        f38 = -x + (1052/res) - m
        f39 = x - (1110/res) - m
        
#Rectangle-11
        f40 = -y + (315/res) - m
        f41 = y - (498/res) - m
        f42 = -x + (438/res) - m
        f43 = x - (529/res) - m
        
#Rectangle-12
        f44 = -y + (265/res) - m
        f45 = y - (341/res) - m
        f46 = -x + (529/res) - m
        f47 = x - (712/res) - m
        
#Rectangle-13
        f48 = -y + (267/res) - m
        f49 = y - (384/res) - m
        f50 = -x + (784.5/res) - m
        f51 = x - (936.5/res) - m

#Equation for boundary 1:
        b1 = y - 1 - m
#Equation for boundary 2:
        b2 = x - 1 - m
#Equation for boundary 3:
        b3 = y - (1010 - 1 - m)
#Equation for boundary 4:
        b4 = x - (1110 - 1 - m)     

        if (c1<=0 or c2<=0 or c3<=0 or c4<=0 or (t1<=0 and t2<=0 and t3<=0 and t4<=0) or t_l<=0 or t_r<=0 or (f1 <= 0 and f2 <= 0 and f3 <= 0 and f4 <= 0) or (f5 <= 0 and f6 <= 0 and f7 <= 0 and f8 <= 0) or (f9 <= 0 and f10 <= 0 and f11 <= 0 and f12 <= 0) or (f13 <= 0 and f14 <= 0 and f15 <= 0 and f16 <= 0) or (f17 <= 0 and f18 <= 0 and f19 <= 0 and f20 <= 0) or (f20_1 <= 0 and f21 <= 0 and f22 <= 0 and f23 <= 0) or (f24 <= 0 and f25 <= 0 and f26 <= 0 and f27 <= 0) or (f28 <= 0 and f29 <= 0 and f30 <= 0 and f31 <= 0) or (f32 <= 0 and f33 <= 0 and f34 <= 0 and f35 <= 0) or (f36 <= 0 and f37 <= 0 and f38 <= 0 and f39 <= 0) or (f40 <= 0 and f41 <= 0 and f42 <= 0 and f43 <= 0) or (f44 <= 0 and f45 <= 0 and f46 <= 0 and f47 <= 0) or (f48 <= 0 and f49 <= 0 and f50 <= 0 and f51 <= 0) or b1<0 or b2<0 or b3>=0 or b4>=0):
            obstacle[x][y] = 1
            ox.append(x)
            oy.append(y)  
            
obstacleList = np.vstack((ox,oy)).T 
plt.plot(ox,oy,"ko")           
plt.axis([-10, 1210, -10, 1110])
plt.grid(True)

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 expandDis=10, goalSampleRate=20, maxIter=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        """
        Pathplanning
        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        for i in range(self.maxIter):
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            newNode = self.steer(rnd, nind)
            #  print(newNode.cost)

            if self.__CollisionCheck(newNode, self.obstacleList):
                nearinds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearinds)
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)

            if animation and i % 5 == 0:
                self.DrawGraph(rnd)

        # generate coruse
        lastIndex = self.get_best_last_index()
        if lastIndex is None:
            return None
        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if not nearinds:
            return newNode

        dlist = []
        for i in nearinds:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            if self.check_collision_extend(self.nodeList[i], theta, d):
                dlist.append(self.nodeList[i].cost + d)
            else:
                dlist.append(float("inf"))

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode.cost = mincost
        newNode.parent = minind

        return newNode

    def steer(self, rnd, nind):

        # expand tree
        nearestNode = self.nodeList[nind]
        theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
        newNode = Node(rnd[0], rnd[1])
        currentDistance = math.sqrt(
            (rnd[1] - nearestNode.y) ** 2 + (rnd[0] - nearestNode.x) ** 2)
        # Find a point within expandDis of nind, and closest to rnd
        if currentDistance <= self.expandDis:
            pass
        else:
            newNode.x = round(nearestNode.x + self.expandDis * math.cos(theta))
            newNode.y = round(nearestNode.y + self.expandDis * math.sin(theta))
        newNode.cost = float("inf")
        newNode.parent = None
        return newNode

    def get_random_point(self):

        if random.randint(0, 1110) > self.goalSampleRate:
            rnd = [round(random.uniform(self.minrand, self.maxrand)),
                   round(random.uniform(self.minrand, self.maxrand))]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y]

        return rnd

    def get_best_last_index(self):

        disglist = [self.calc_dist_to_goal(
            node.x, node.y) for node in self.nodeList]
        goalinds = [disglist.index(i) for i in disglist if i <= self.expandDis]

        if not goalinds:
            return None

        mincost = min([self.nodeList[i].cost for i in goalinds])
        for i in goalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append([node.x, node.y])
            goalind = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, newNode):
        nnode = len(self.nodeList)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        #  r = self.expandDis * 5.0
        dlist = [(node.x - newNode.x) ** 2 +
                 (node.y - newNode.y) ** 2 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def rewire(self, newNode, nearinds):
        nnode = len(self.nodeList)
        for i in nearinds:
            nearNode = self.nodeList[i]

            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            d = math.sqrt(dx ** 2 + dy ** 2)

            scost = newNode.cost + d

            if nearNode.cost > scost:
                theta = math.atan2(dy, dx)
                if self.check_collision_extend(nearNode, theta, d):
                    nearNode.parent = nnode - 1
                    nearNode.cost = scost

    def check_collision_extend(self, nearNode, theta, d):

        tmpNode = copy.deepcopy(nearNode)

        for i in range(int(d / self.expandDis)):
            tmpNode.x += self.expandDis * math.cos(theta)
            tmpNode.y += self.expandDis * math.sin(theta)
            if not self.__CollisionCheck(tmpNode, self.obstacleList):
                return False

        return True

    def DrawGraph(self, rnd=None):
        """
        Draw Graph
        """
        #plt.clf()
        #o_x = [ox for ox,oy in self.obstacleList]
        #o_y = [oy for ox,oy in self.obstacleList]
# =============================================================================
#         if rnd is not None:
#             plt.plot(rnd[0], rnd[1], "^k")
# =============================================================================
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")
            
            
# =============================================================================
#         for (ox, oy, size) in self.obstacleList:
#             plt.plot(ox, oy, "ok", ms=30 * size)
# =============================================================================

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        # plt.plot(o_x,o_y,"ko")
        plt.axis([-10, 1210, -10, 1110])
        plt.grid(True)
        plt.pause(0.001)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node, obstacleList):
        for ox, oy in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy
            if d <= 20:
                return False  # collision

        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start " + __file__)
    

# =============================================================================
#     # ====Search Path with RRT====
#     obstacleList = [
#         (5, 5, 1),
#         (3, 6, 2),
#         (3, 8, 2),
#         (3, 10, 2),
#         (7, 5, 2),
#         (9, 5, 2)
#     ]  # [x,y,size(radius)]
# 
# =============================================================================
    # Set Initial parameters
    rrt = RRT(start=[400, 400], goal=[600, 400],
              randArea=[0, 1110], obstacleList=obstacleList)
    path = rrt.Planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.DrawGraph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
