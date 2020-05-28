# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:13:13 2019

@author: nakul
"""

import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 expandDis= 10, goalSampleRate=20, maxIter=500):
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
                self.DrawGraph(self.obstacleList, rnd)              # Change

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
            newNode.x = nearestNode.x + self.expandDis * math.cos(theta)
            newNode.y = nearestNode.y + self.expandDis * math.sin(theta)
        newNode.cost = float("inf")
        newNode.parent = None
        return newNode

    def get_random_point(self):

        if random.randint(0, 1110) > self.goalSampleRate:              # Change
            rnd = [random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand)]
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

    def DrawGraph(self, obstacleList, rnd=None):               
        """
        Draw Graph
        """
        plt.clf()
        o_x = [x[0] for x in obstacleList]                      # Change
        o_y = [y[1] for y in obstacleList]                      # Change
        
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")

#==============================================================================
#         for (ox, oy) in self.obstacleList:
#             plt.plot(ox, oy, "ok", ms=30)
#==============================================================================

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.plot(o_x, o_y, "ko")                                  # Change
        #plt.axis([-2, 15, -2, 15])
        plt.axis([0, 1110, 0, 1010])                                # Change
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node, obstacleList):
        for i in obstacleList:
            dx = i[0] - node.x                                   # Change
            dy = i[1] - node.y                                   # Change
            d = dx * dx + dy * dy
            if d <= 2:
                return False  # collision

        return True  # safe



#====================================================================

def check_boundary(x, y, r):
    bool1= (x >= 0 and x <= r ) or ( x >= 1110-r and x <= 1110 ) 
    bool2= (y >= 0 and y <= r ) or ( y >= 1010-r and y <= 1010 ) 
    req = False    
    if (bool1 or bool2):
        req = True
    return req
# For rectangles
def check_rect(x, y, r):
    #rect1
    f1 = x - 918 - r
    f2 = 832 - x - r
    f3 = 827 - y  - r 
    f4 = y - 1070 - r
    rec1 = (f1 <= 0 and f2 <= 0 and f3 <=0 and f4 <= 0)
    
    #rect2
    f1_2 = x - 1026 - r
    f2_2 = 983 - x - r
    f3_2 = 919 - y - r
    f4_2 = y - 1010 - r 
    rec2 = (f1_2 <= 0 and f2_2 <= 0 and f3_2 <=0 and f4_2 <= 0)
    
    #rect3
    f1_3 = x - 1110 - r
    f2_3 = 744 - x - r
    f3_3 = 621 - y - r
    f4_3 = y - 697 - r
    rec3 = (f1_3 <= 0 and f2_3 <= 0 and f3_3 <=0 and f4_3 <= 0)
    
    #rect4
    f1_4 = x - 1110 - r
    f2_4 = 1052 - x - r
    f3_4 = 448.5 - y - r
    f4_4 = y - 565.5 - r
    rec4 = (f1_4 <= 0 and f2_4 <= 0 and f3_4 <=0 and f4_4 <= 0)
    
    #rect5
    f1_5 = x - 1110 - r
    f2_5 = 1019 - x - r
    f3_5 = 362.5 - y - r
    f4_5 = y - 448.5 - r
    rec5 = (f1_5 <= 0 and f2_5 <= 0 and f3_5 <=0 and f4_5 <= 0)
    
    #rect6
    f1_6 = x - 1110 - r
    f2_6 = 1052 - x - r
    f3_6 = 178.25 - y - r
    f4_6 = y - 295.25 - r
    rec6 = (f1_6 <= 0 and f2_6 <= 0 and f3_6 <=0 and f4_6 <= 0)
    
    #rect7
    f1_7 = x - 1110 - r
    f2_7 = 927 - x - r
    f3_7 = 35 - y - r
    f4_7 = y - 111 - r
    rec7 = (f1_7 <= 0 and f2_7 <= 0 and f3_7 <=0 and f4_7 <= 0)
    
    #rect8 
    f1_8 = x - 1110 - r 
    f2_8 = 685 - x - r
    f3_8 = 0 - y - r
    f4_8 = y - 35 - r
    rec8 = (f1_8 <= 0 and f2_8 <= 0 and f3_8 <=0 and f4_8 <= 0)
    
    #rect9   
    f1_9 = x - 896 - r
    f2_9 = 779 - x - r
    f3_9 = 35 - y - r
    f4_9 = y - 93 - r
    rec9 = (f1_9 <= 0 and f2_9 <= 0 and f3_9 <=0 and f4_9 <= 0)
    
    #rect10
    f1_10 = x - 748 - r
    f2_10 = 474 - x - r
    f3_10 = 35 - y - r
    f4_10 = y - 187 - r
    rec10 = (f1_10 <= 0 and f2_10 <= 0 and f3_10 <=0 and f4_10 <= 0)
    
    #rect11
    f1_11 = x - 712 - r
    f2_11 = 529 - x - r
    f3_11 = 265 - y - r
    f4_11 = y - 341 - r
    rec11 = (f1_11 <= 0 and f2_11 <= 0 and f3_11 <=0 and f4_11 <= 0)
    
    #rect12
    f1_12 = x - 529 - r 
    f2_12 = 438 - x - r
    f3_12 = 315 - y - r
    f4_12 = y - 498 - r
    rec12 = (f1_12 <= 0 and f2_12 <= 0 and f3_12 <=0 and f4_12 <= 0)
    
    #rect13 
    f1_13 = x - 936.5 - r
    f2_13 = 784.5 - x - r
    f3_13 = 267 - y - r
    f4_13 = y - 384 - r
    rec13 = (f1_13 <= 0 and f2_13 <= 0 and f3_13 <=0 and f4_13 <= 0)
    
    req= False 
    if (rec1 or rec2 or rec3 or rec4 or rec5 or rec6 or rec7 or rec8 or rec9 
        or rec10 or rec11 or rec12 or rec13):
        req = True
    return req  
# For circles
def check_circle(x, y, r):
    eqn_circle_1= (x - 390)**2 + (y - 965)**2 - (40.5 + r)**2
    eqn_circle_2= (x - 438)**2 + (y - 736)**2 - (40.5 + r)**2
    eqn_circle_3= (x - 390)**2 + (y - 45)**2 - (40.5 + r)**2
    eqn_circle_4= (x - 438)**2 + (y - 274)**2 - (40.5 + r)**2    
    req = False
    # using semi-algabraic equation to define obstacle space
    if (eqn_circle_1 <= 0 or eqn_circle_2 <= 0 or eqn_circle_3 <= 0 or eqn_circle_4 <= 0):
        req = True
    return req
# For ellipse
def check_ellipse(x, y, r):
    sq_1 = x - 310 
    sq_2 = 150 - x 
    sq_3 = 750 - y - r
    sq_4 = y - 910 - r
    bool1 = (sq_1 <= 0 and sq_2 <= 0 and sq_3 <= 0 and sq_4 <= 0)
    #r1
    eq_circle_1 = (x - 150)**2 + (y - 830)**2 - (80 + r)**2
    #r2     
    eq_circle_2 = (x - 310)**2 + (y - 830)**2 - (80 + r)**2
    req = False
    if (bool1 or eq_circle_1 <=0 or eq_circle_2 <=0):
        req = True
    return req

#====================================================================


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

    # ====Search Path with RRT====
   
    ox=[]                                                       # Change
    oy=[]
    obstacleList=[]
    #robot radius(in cm)
    r = 30 
    for i in range(0,1111):
        for j in range(0,1011):
            req0 = check_boundary(i, j, r)
            req1 = check_rect(i, j, r)
            req2 = check_circle(i, j, r)
            req3 = check_ellipse(i, j, r)
                 
            if (req0 or req1 or req2 or req3):
                ox.append(i)
                oy.append(j)
                obstacleList.append((i,j))                   # Change
    
    #print("Obstacle: ", obstacleList)
    
#==============================================================================
#     obstacleList = [
#         (5, 5, 1),
#         (3, 6, 2),
#         (3, 8, 2),
#         (3, 10, 2),
#         (7, 5, 2),
#         (9, 5, 2)
#     ]  # [x,y,size(radius)]
#==============================================================================

    # Set Initial parameters
    rrt = RRT(start=[400,400], goal=[600, 400],
              randArea=[0, 1110], obstacleList=obstacleList)
    path = rrt.Planning(animation=show_animation)

    if path is None:
        print("\n Cannot find path.")
    else:
        print("\n Path found!!")

        # Draw final path
        if show_animation:
            rrt.DrawGraph(obstacleList)                          # Change
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r',linewidth=2)
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
    
    
    