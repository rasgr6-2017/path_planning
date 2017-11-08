#!/usr/bin/env python
import roslib
import sys
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import time
import math
from scipy import weave
# import bezier
from scipy.interpolate import splev, splrep


class AStarNode:
    def __init__(self, index, parent, g, h):
        self.index = index
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h


class RoadGraph:
    def __init__(self, all_nodes):
        self.edge_dict = dict()
        self.edge_list = list()
        self.connection = list()
        self.nodes = all_nodes

    def add_edge(self, edge_key, edge_value):
        """edge_key is a tuple or pair; edge_value is a list of pair"""
        self.edge_list.append(edge_value)
        self.edge_dict[edge_key] = (edge_value.__len__(), edge_list.__len__()-1)
        self.connection.append(edge_key)

    def check_connection(self, node_1, node_2):
        key_for = (node_1, node_2)
        key_back = (node_2, node_1)

        if key_for in self.connection:
            return True
        elif key_back in self.connection:
            return True
        else:
            return False

    def get_edge(self, key):
        key_inv = (key[1], key[0])
        if key in self.connection:
            tag = self.edge_dict[key][1]
            return self.edge_list[tag]
        elif key_inv in self.connection:
            tag = self.edge_dict[key_inv][1]
            return self.edge_list[tag]
        else:
            empty_list = []
            print("Non-existing edge! Returning empty list!")
            return empty_list

    def node_to_node(self, node_1, node_2):
        """find whether there is an edge between node_1 and node_2"""
        key_for = (node_1, node_2)
        key_back = (node_2, node_1)
        """search direct link from node_1 to node_2"""
        if key_for in self.connection:
            cost = self.edge_dict[key_for][0]
        elif key_back in self.connection:
            cost = self.edge_dict[key_back][0]
        else:
            cost = 99999
        return cost

    def heuristic(self, node_1, node_2):
        (x1, y1) = self.nodes[node_1]
        (x2, y2) = self.nodes[node_2]

        """heuristic is suspicious"""
        euclidean = int(math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2)) / 1.414)
        return euclidean

    def get_neighbour(self, node_a):
        neighbour = list()
        for i in range(len(self.nodes)):
            key_for = (node_a, i)
            key_back = (i, node_a)
            if key_for in self.connection or key_back in self.connection:
                neighbour.append(i)

        return neighbour

    def a_star_path(self, node_1, node_2):
        """node_1 and node_2 are number of node, not the coordinate"""
        closed_node = list()

        n1 = AStarNode(node_1, -1, 0, self.heuristic(node_1, node_2))

        open_node = list()
        open_node.append(n1)

        while open_node.__len__() > 0:
            min_f = 999999
            n_q = -1
            for n_i in range(len(open_node)):
                f_temp = open_node[n_i].f
                if f_temp < min_f:
                    min_f = f_temp
                    n_q = n_i

            node_q = open_node.pop(n_q)

            if min_f == 99999:
                print("no solution?!")
                return

            successor = self.get_neighbour(node_q.index)
            for each in successor:

                node_temp = AStarNode(each, node_q.index, 0, 0)

                if each == node_2:
                    print("found path")
                    return self.get_path(closed_node, node_q, node_temp)

                node_temp.g = node_q.g + self.node_to_node(node_q.index, each)
                node_temp.h = self.heuristic(each, node_2)
                node_temp.f = node_temp.g + node_temp.h

                open_flag = True
                closed_flag = True

                for node_in_open in open_node:
                    if node_in_open.index == node_temp.index and node_in_open.f <= node_temp.f:
                        open_flag = False
                for node_in_closed in closed_node:
                    if node_in_closed.index == node_temp.index and node_in_closed.f <= node_temp.f:
                        closed_flag = False

                # if this node is not definitely useless, we add it to open_node list
                if open_flag and closed_flag:
                    open_node.append(node_temp)

            # if a node's neighbour are checked, then we put it into closed_node list
            closed_node.append(node_q)

    def get_path(self, closed_nodes, node_q, node_end):
        path_edges = list()
        path_index = list()
        pair_temp = (node_q.index, node_end.index)
        path_edges.append(self.get_edge(pair_temp))
        path_index.append(pair_temp)

        current_node = node_q
        while current_node.parent != -1:
            for node_left in closed_nodes:
                if node_left.index == current_node.parent:
                    pair_temp = (node_left.index, current_node.index)
                    path_edges.insert(0, self.get_edge(pair_temp))
                    path_index.insert(0, pair_temp)
                    current_node = node_left
                    break
        return path_edges, path_index


def white_to_black(a, b):

    # from white to black is from 1 to 0
    if (a == 1 or a == 255) and b == 0:
        return 1
    else:
        return 0


def get_step(img, i, j, number):
    """used for Zhang Suen thinning"""
    # if i == 100 and j == 100:
    #    print "debug"

    p2 = int(img[i-1][j])
    p3 = int(img[i-1][j+1])
    p4 = int(img[i][j+1])
    p5 = int(img[i+1][j+1])
    p6 = int(img[i+1][j])
    p7 = int(img[i+1][j-1])
    p8 = int(img[i][j-1])
    p9 = int(img[i-1][j-1])
    
    b = int(8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9))

    # print b
    
    if b < 2 or b > 6:
        return False

    a = int( (p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0) )

    
    if a != 1:
        return False

    if number == 1:
        if (p2 or p4 or p6) and (p4 or p6 or p8):
            acc_3 += time.time() - t_t
            return True
    else:
        if (p2 or p4 or p8) and (p2 or p6 or p8):
            acc_3 += time.time() - t_t
            return True

    return False
    

def zs_thinning(img):
    """doing Zhang Suen thinning"""
    image = img.copy()

    # this to a degree can represent the expansion of walls and obstacles
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image = cv2.erode(image, kernel1, iterations=2)

    image = cv2.bitwise_not(image)

    # image = cv2.medianBlur(image, 3)

    # plt.matshow(image, 5)
    # plt.show()

    # print image[100]

    # cv2.imshow("test", img)

    row, col = image.shape[:2]

    time_acc = 0.
    time_acc_2 = 0.
    
    global acc_1
    global acc_2
    global acc_3
    global acc_4
	
    image[:, :] = image[:, :] / 255
    
    a = b = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = int(1)

    ans = False

    while True:
        turn1 = []
        turn2 = []
        t_1 = time.time()
        for i in range(1, row-1):
            for j in range(1, col-1):
                t_t = time.time()
            	acc_4 += time.time() - t_1
                if (image[i][j]==0):
                    t_1 = time.time()
                    ans = False
                    p2 = int(image[i-1][j])
                    p3 = int(image[i-1][j+1])
                    p4 = int(image[i][j+1])
                    p5 = int(image[i+1][j+1])
                    p6 = int(image[i+1][j])
                    p7 = int(image[i+1][j-1])
                    p8 = int(image[i][j-1])
                    p9 = int(image[i-1][j-1])
                    b = int(8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9))
                    acc_1 += time.time() - t_1
                    
                    if 2 <= b <= 6:
                        t_1 = time.time()
                        a = int( (p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0) )
                        acc_2 += time.time() - t_1
                        if a == 1:
                            t_1 = time.time()
                            if (p2 or p4 or p6) and (p4 or p6 or p8):
                                ans = True
                            acc_3 += time.time() - t_1
                    if ans:              
                          
                        turn1.append((i, j))
                        
                t_1 = time.time()            
                time_acc_2 += time.time()-t_t
                    
        for x, y in turn1:
            image[x][y] = 1  # set condition1 satisfying pixel to white

        t_1 = time.time()
        for i in range(1, row-1):
            for j in range(1, col-1):
                acc_4 += time.time() - t_1
                t_t = time.time()
                if (image[i][j]==0):
                    t_1 = time.time()
                    ans = False
                    p2 = int(image[i-1][j])
                    p3 = int(image[i-1][j+1])
                    p4 = int(image[i][j+1])
                    p5 = int(image[i+1][j+1])
                    p6 = int(image[i+1][j])
                    p7 = int(image[i+1][j-1])
                    p8 = int(image[i][j-1])
                    p9 = int(image[i-1][j-1])   
                    b = 8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)
                    acc_1 += time.time() - t_1
                    
                    if 2 <= b <= 6:
                        t_1 = time.time()
                        
                        a = int( (p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0) )
                        acc_2 += time.time() - t_1
                        if a == 1:
                            t_1 = time.time()
                            if (p2 or p4 or p8) and (p2 or p6 or p8):
                                ans = True
                            acc_3 += time.time() - t_1
                    if ans:                    
                        turn2.append((i, j)) 
                        
                t_1 = time.time() 
                time_acc += time.time()-t_t

        for p, q in turn2:
            image[p][q] = 1  # set condition satisfying pixel to white

        # print " turn1 " + str(len(turn1)) + " turn2 " + str(len(turn2)) + " time 1 " + str(time_acc) + " time 2 " + str(time_acc_2)
        # print " acc1 " + str(acc_1) + " acc2 " + str(acc_2) + " acc3 " + str(acc_3) + " acc4 " + str(acc_4)
        # plt.matshow(image, 5)
        # plt.show()
        
        if len(turn1) == 0 and len(turn2) == 0:
            break

    return image


def is_neighbour(pair1, pair2):
    """check whether two point(represented by pair) are neighbour or not"""
    neighbour_range = 2
    if -neighbour_range <= pair1[0] - pair2[0] <= neighbour_range and \
            -neighbour_range <= pair1[1] - pair2[1] <= neighbour_range:
        return True
    else:
        return False


def path_any_point(road_graph, map_img, point_1, point_2):
    """find path between any two point in map"""
    temp_map = map_img.copy()

    # this to a degree can represent the expansion of walls and obstacles
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    temp_map = cv2.erode(temp_map, kernel1, iterations=2)

    if temp_map[point_1[0]][point_1[1]] == 0 or temp_map[point_2[0]][point_2[1]] == 0:
        print("invalid starting or ending position! You are colliding the wall there!")
        empty_list = []
        return empty_list, empty_list

    # open_list = list()
    # closed_list = list()
    # current_point = point_1

    min_1 = 999999
    min_2 = 999999
    index_1 = [0, 0]
    index_2 = [0, 0]
    for i in range(road_graph.edge_list.__len__()):
        edge_i = road_graph.edge_list[i]
        for j in range(edge_i.__len__()):
            point_j = edge_i[j]
            dist_1 = diagonal_distance(point_j, point_1)
            dist_2 = diagonal_distance(point_j, point_2)
            if dist_1 < min_1:
                min_1 = dist_1
                index_1 = [i, j]
            if dist_2 < min_2:
                min_2 = dist_2
                index_2 = [i, j]

    # following process is toooooo clumsy
    node_11 = road_graph.connection[index_1[0]][0]
    node_12 = road_graph.connection[index_1[0]][1]
    node_21 = road_graph.connection[index_2[0]][0]
    node_22 = road_graph.connection[index_2[0]][1]
    path_1, ind1 = road_graph.a_star_path(node_11, node_21)
    path_2, ind2 = road_graph.a_star_path(node_11, node_22)
    path_3, ind3 = road_graph.a_star_path(node_12, node_21)
    path_4, ind4 = road_graph.a_star_path(node_12, node_22)

    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 0
    for item in path_1:
        l1 = l1 + item.__len__()
    for item in path_2:
        l2 = l2 + item.__len__()
    for item in path_3:
        l3 = l3 + item.__len__()
    for item in path_4:
        l4 = l4 + item.__len__()

    if l1 < l2 and l1 < l3 and l1 < l4:
        i = index_1[0]
        j = index_1[1]
        path_1.insert(0, road_graph.edge_list[i])
        ind1.insert(0, tuple(road_graph.edge_list[i][j]))
        i = index_2[0]
        j = index_2[1]
        path_1.append(road_graph.edge_list[i])
        ind1.append(tuple(road_graph.edge_list[i][j]))
        return path_1, ind1
    elif l2 < l1 and l2 < l3 and l2 < l4:
        i = index_1[0]
        j = index_1[1]
        path_2.insert(0, road_graph.edge_list[i])
        ind2.insert(0, tuple(road_graph.edge_list[i][j]))
        i = index_2[0]
        j = index_2[1]
        path_2.append(road_graph.edge_list[i])
        ind2.append(tuple(road_graph.edge_list[i][j]))
        return path_2, ind2
    elif l3 < l1 and l3 < l2 and l3 < l4:
        i = index_1[0]
        j = index_1[1]
        path_3.insert(0, road_graph.edge_list[i])
        ind3.insert(0, tuple(road_graph.edge_list[i][j]))
        i = index_2[0]
        j = index_2[1]
        path_3.append(road_graph.edge_list[i])
        ind3.append(tuple(road_graph.edge_list[i][j]))
        return path_3, ind3
    else:
        i = index_1[0]
        j = index_1[1]
        path_4.insert(0, road_graph.edge_list[i])
        ind4.insert(0, tuple(road_graph.edge_list[i][j]))
        i = index_2[0]
        j = index_2[1]
        path_4.append(road_graph.edge_list[i])
        ind4.append(tuple(road_graph.edge_list[i][j]))
        return path_4, ind4


def diagonal_distance(point_a, point_b):
    dx = abs(point_a[0] - point_b[0])
    dy = abs(point_a[1] - point_b[1])
    if dx > dy:
        return dx
    else:
        return dy


def mid_point(a1, a2):
    b = np.array(np.zeros(a1.size), dtype=a1.dtype)
    for i in range(a1.size):
        b[i] = (a1[i] + a2[i]) / 2.0
    return b


if __name__ == "__main__":

    # get an image representing the obstacles and walls
    img_list = ["maze2.png", "maze2_more.png", "maze2_less.png"]
    img_path = "/home/ras16/catkin_ws/src/path_planning/scripts/"
    
    for img_n in range(1):

        original = cv2.imread(img_path + img_list[img_n])

        height, width = original.shape[:2]

        # resize the image to get lower resolution
        # this makes computational cost decrease much
        original = cv2.resize(original, (int(0.3*width), int(0.3*height)), interpolation=cv2.INTER_CUBIC)
        print "image height" + str(int(0.3*height)) + "image width" + str(int(0.3*width))

        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

        # use only binary map
        # with probabilistic occupancy, the threshold of this should be set carefully
        ret3, my_map_original = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        acc_1 = 0.
        acc_2 = 0.
        acc_3 = 0.
        acc_4 = 0.

        t_start = time.time()
        # do erosion first, and do thinning
        # this Zhang Suen thinning guarantees that skeleton will not be cut at any point
        thinning = zs_thinning(my_map_original)
        t_end = time.time()
        print (t_end-t_start)

        joint_count = 0

        node_list = []

        # find the joint node of branching point
        th, tw = thinning.shape[:2]
        for pi in range(1, th-1):
            for pj in range(1, tw-1):
                if thinning[pi][pj] == 0:
                    p2 = thinning[pi - 1][pj]
                    p3 = thinning[pi - 1][pj + 1]
                    p4 = thinning[pi][pj + 1]
                    p5 = thinning[pi + 1][pj + 1]
                    p6 = thinning[pi + 1][pj]
                    p7 = thinning[pi + 1][pj - 1]
                    p8 = thinning[pi][pj - 1]
                    p9 = thinning[pi - 1][pj - 1]

                    black_n = (8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9))

                    if 1 < black_n < 3 or black_n > 5:
                        continue

                    if (white_to_black(p2, p3) + white_to_black(p3, p4) +
                            white_to_black(p4, p5) + white_to_black(p5, p6) +
                            white_to_black(p6, p7) + white_to_black(p7, p8) +
                            white_to_black(p8, p9) + white_to_black(p9, p2) >= 3):
                        joint_count = joint_count + 1
                        thinning[pi][pj] = 150
                        node_list.append((pi, pj))

                    if black_n == 1:
                        thinning[pi][pj] = 150
                        joint_count = joint_count + 1
                        node_list.append((pi, pj))

        voronoi_graph = RoadGraph(node_list)

        # print joint_count
        # print node_list

        # to show the node with a larger square dot
        # in labelling of connected components,
        # it is also used as edge cutting
        for pi, pj in node_list:
            for ia in range(-1, 2):
                for ja in range(-1, 2):
                    thinning[pi + ia][pj + ja] = 1

        # to label all the connected component
        res, label_num = label(thinning, neighbors=8, background=1, return_num=True)
        res = res + 1
        # print res
        props = regionprops(res)
        
        print "label number " + str(label_num) + "  " + str(len(props))
        
        # plt.matshow(res, img_n)
        # plt.show()
        # plt.pause(10)

        # to show the walls in original map which is removed from the map
        for pi in range(1, th-1):
            for pj in range(1, tw-1):
                if my_map_original[pi][pj] == 0:
                    res[pi][pj] = label_num + 2

        # in edge list, we have label_num edges, made up by a list of coordinate pairs
        edge_list = dict()
        for l_i in range(label_num):
            temp_list = props[l_i].coords
            temp_pair = []
            for point in temp_list:
                for node_i in node_list:
                    if is_neighbour(point, node_i):
                        if not (node_list.index(node_i) in temp_pair):
                            temp_pair.append(node_list.index(node_i))
                            temp_array = np.array([[node_i[0], node_i[1]]])
                            temp_list = np.concatenate((temp_list, temp_array), axis=0)

            # if the edge does not link two different nodes
            # then it must be a loop since it is detected as an edge
            if len(temp_pair) == 1:
                temp_pair.append(temp_pair[0])
                
            edge_list[(temp_pair[0], temp_pair[1])] = temp_list
            voronoi_graph.add_edge((temp_pair[0], temp_pair[1]), temp_list)

        t_start = time.time()
        """let's build a graph and find path now!"""
        # path_found, node_index = voronoi_graph.a_star_path(1, 15)
        # print node_index

        """let's try find path between any two point!"""
        path2, pd2 = path_any_point(voronoi_graph, my_map_original, (10, 10), (85, 10))
        """in pd2, the first and last element are just cutting point into some edge"""
        if len(pd2) != 0:
            print pd2
        # cv2.imshow("label" + str(img_n), res)

        # test_nodes = np.array([[60, 60], [64, 59], [67, 55], [72, 55], [74, 52], [78, 53], [82, 56], [83, 60]],
        #                      np.double)

        test_nodes = np.array([[60, 60], [64, 59], [68, 60], [73, 60], [76, 62], [78, 65], [79, 69], [79, 74]],
                              np.double)

        """
        p_01 = mid_point(test_nodes[0], test_nodes[1])
        p_12 = mid_point(test_nodes[1], test_nodes[2])
        p_23 = mid_point(test_nodes[2], test_nodes[3])
        p_34 = mid_point(test_nodes[3], test_nodes[4])
        p_45 = mid_point(test_nodes[4], test_nodes[5])
        p_56 = mid_point(test_nodes[5], test_nodes[6])
        p_67 = mid_point(test_nodes[6], test_nodes[7])

        curve_1 = bezier.Curve(np.array([p_01, test_nodes[1], p_12], dtype=np.double), degree=3)
        curve_2 = bezier.Curve(np.array([p_12, test_nodes[2], p_23], dtype=np.double), degree=3)
        curve_3 = bezier.Curve(np.array([p_23, test_nodes[3], p_34], dtype=np.double), degree=3)
        curve_4 = bezier.Curve(np.array([p_34, test_nodes[4], p_45], dtype=np.double), degree=3)
        curve_5 = bezier.Curve(np.array([p_45, test_nodes[5], p_56, test_nodes[6], p_67], dtype=np.double), degree=5)
        # curve_6 = bezier.Curve(np.array([p_56, test_nodes[6], p_67], dtype=np.double), degree=3)

        s_values = np.linspace(0.0, 1.0, 20)

        fit_1 = curve_1.evaluate_multi(s_values)
        fit_2 = curve_2.evaluate_multi(s_values)
        fit_3 = curve_3.evaluate_multi(s_values)
        fit_4 = curve_4.evaluate_multi(s_values)
        fit_5 = curve_5.evaluate_multi(s_values)
        # fit_6 = curve_6.evaluate_multi(s_values)

        plt.plot(fit_1[:, 0], fit_1[:, 1], 'b--',  fit_2[:, 0], fit_2[:, 1], 'g--',
                 fit_3[:, 0], fit_3[:, 1], 'r--',  fit_4[:, 0], fit_4[:, 1], 'k--',
                 fit_5[:, 0], fit_5[:, 1], 'c--',
                 test_nodes[:,0], test_nodes[:,1], 'ro')
        # fit_6[:, 0], fit_6[:, 1], 'm--',
        """

        """spline interplotation"""
        # sp1 = splrep(test_nodes[:, 0], test_nodes[:, 1], k=4)
        # x_value = np.linspace(60, 83, 200)
        # y_value = splev(x_value, sp1)
        # plt.plot(x_value, y_value, 'b--', test_nodes[:,0], test_nodes[:,1])

        """bezier python library"""
        # b_curve = bezier.Curve(test_nodes.astype(dtype=np.double), degree=8)

        # s_values = np.linspace(0.0, 1.0, 20)

        # fit_value = b_curve.evaluate_multi(s_values)

        # plt.plot(fit_value[:, 0], fit_value[:, 1], 'b--', test_nodes[:,0], test_nodes[:,1], 'ro')

        # fit the edge segment with polynomial is not good
        """
        poly_para = []
        for key_i, list_i in edge_list.iteritems():
            y = list_i[:, 0]
            x = list_i[:, 1]
            poly_temp = np.polyfit(list_i[:, 1], list_i[:, 0], )
            poly_para.append(poly_temp)
            poly = np.poly1d(poly_temp)
            start_x = node_list[key_i[0]][1]
            start_y = node_list[key_i[0]][0]
            end_x = node_list[key_i[1]][1]
            end_y = node_list[key_i[1]][0]

            xp = np.linspace(start_x, 0.1, end_x)
            _ = plt.plot(x, y, '.', xp, poly(xp), '-')
        """

        # thinning = cv2.imread('thinning.png')
        # thinning = cv2.cvtColor(thinning, cv2.COLOR_RGB2GRAY)
        # ret3, thinning = cv2.threshold(thinning, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imwrite('thinning' + str(img_n) + '.png', thinning)

        # cv2.imshow("original", my_map)
        # cv2.imshow("thinning", thinning)

        # print thinning.shape[:2]
        # print my_map.shape[:2]

        # test = cv2.bitwise_and(original, original, mask=thinning)
        # cv2.imshow("eroded" + str(img_n), my_map)
        # cv2.imshow("test" + str(img_n), test)

        # cv2.imwrite('result' + str(img_n) + '.png', test)

        # print thinning[20]
	print(t_start - t_end)
    cv2.waitKey(0)
