#!/usr/bin/env python
import roslib
import sys
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import time
import math
import multiprocessing



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

    def get_length(self, key):
        key_inv = (key[1], key[0])
        if key in self.connection:
            return self.edge_dict[key][0]
        elif key_inv in self.connection:
            return self.edge_dict[key_inv][0]
        else:
            print("Non-existing edge! Returning zero!")
            return 0

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

        if node_1 == node_2:
            print "Warning: two nodes are the same!"

        for con in self.connection:
            if con == tuple([node_1, node_2]) or con == tuple([node_2, node_1]):
                print "Warning: directly connected nodes!"
                return self.get_edge(con), con

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

    def pixel_path(self, p1, p_2, ps=None, pe=None, interval=16):
        """p1, p_2 are connected nodes in the order you want to go; p are optional point where you are"""
        if ps is None:
            ps = self.nodes[p1]

        if pe is None:
            pe = self.nodes[p_2]
        point1 = list(ps)
        point2 = list(pe)
        temp_edge = list(self.get_edge((p1, p_2)))
        cut1 = temp_edge.index(point1)
        cut2 = temp_edge.index(point2)
        if cut1 <= cut2:
            temp_edge = temp_edge[cut1:(cut2+1)]
        else:
            temp_edge = temp_edge[cut2:(cut1+1)]
        pixel_path = list()
        pixel_path.append(point1)
        current_pixel = point1
        temp_edge.remove(point1)
        min_dist = 9999.0
        while temp_edge.__len__() > 0:
            for e_i in xrange(len(temp_edge) - 1, -1, -1):
                pixel = temp_edge[e_i]
                di = diagonal_distance(current_pixel, pixel)
                if di < interval:
                    if pixel == point2:
                        pixel_path.append(point2)
                        return pixel_path
                    else:
                        temp_edge.remove(pixel)
                    continue
                else:
                    if di < min_dist:
                        min_dist = di
                        candidate = pixel
            pixel_path.append(candidate)
            if candidate == point2:
                return pixel_path
            current_pixel = candidate
            min_dist = 9999.0
            temp_edge.remove(candidate)
        return pixel_path
        print "Warning: due to some reason, end node is not in the pixel path"


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

    a = int((p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0))
    
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


def scanner_1(image, row_s, row_e):
    t1 = []

    # print "scan 1 " + str(row_s)
    row, col = image.shape[:2]
    for i in range(row_s, row_e):
        for j in range(1, col - 1):
            if image[i][j] == 0:
                ans = False
                p2 = int(image[i - 1][j])
                p3 = int(image[i - 1][j + 1])
                p4 = int(image[i][j + 1])
                p5 = int(image[i + 1][j + 1])
                p6 = int(image[i + 1][j])
                p7 = int(image[i + 1][j - 1])
                p8 = int(image[i][j - 1])
                p9 = int(image[i - 1][j - 1])
                b = int(8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9))
                if 2 <= b <= 6:
                    a = int(
                        (p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (
                        p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0))
                    if a == 1:
                        if (p2 or p4 or p6) and (p4 or p6 or p8):
                            ans = True
                if ans:
                    t1.append((i, j))

    # print "t1 " + str(t1.__len__())
    return t1


def scanner_2(image, row_s, row_e):
    t2 = []
    # print "scan 2 " + str(row_s)
    row, col = image.shape[:2]
    for i in range(row_s, row_e):
        for j in range(1, col - 1):
            if image[i][j] == 0:
                ans = False
                p2 = int(image[i - 1][j])
                p3 = int(image[i - 1][j + 1])
                p4 = int(image[i][j + 1])
                p5 = int(image[i + 1][j + 1])
                p6 = int(image[i + 1][j])
                p7 = int(image[i + 1][j - 1])
                p8 = int(image[i][j - 1])
                p9 = int(image[i - 1][j - 1])
                b = int(8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9))
                if 2 <= b <= 6:
                    a = int(
                        (p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (
                        p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0))
                    if a == 1:
                        if (p2 or p4 or p8) and (p2 or p6 or p8):
                            ans = True
                if ans:
                    t2.append((i, j))

    # print "t2 " + str(t2.__len__())
    return t2


def zs_thinning(img):
    """doing Zhang Suen thinning"""
    image = img.copy()

    # this to a degree can represent the expansion of walls and obstacles
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.erode(image, kernel1, iterations=3)

    image = cv2.bitwise_not(image)

    row, col = image.shape[:2]
    
    # plt.matshow(image, 1)
    # plt.show()
    # plt.pause(60)

    image[:, :] = image[:, :] / 255

    count = [0, 0]

    row, col = image.shape[:2]

    # create pools to run multiprocesses
    pool1 = multiprocessing.Pool(processes=4)
    pool2 = multiprocessing.Pool(processes=4)

    row_pairs = [[1, row / 4], [row / 4, row / 2], [row / 2, 3 * row / 4], [3 * row / 4, row - 1]]


    result1 = []
    result2 = []
    for i in range(4):
        result1.append(pool1.apply_async(scanner_1, (image, row_pairs[i][0], row_pairs[i][1])))

    # plt.matshow(image, 5)
    # plt.show()

    while True:

        for x, y in result1[0].get():
            image[x][y] = 1  # set condition1 satisfying pixel to white
        for x, y in result1[1].get():
            image[x][y] = 1      
        for x, y in result1[2].get():
            image[x][y] = 1
        for x, y in result1[3].get():
            image[x][y] = 1


        for i in range(4):
            result2.append(pool2.apply_async(scanner_2, (image, row_pairs[i][0], row_pairs[i][1])))

        for x, y in result2[0].get():
            image[x][y] = 1  # set condition2 satisfying pixel to white
        for x, y in result2[1].get():
            image[x][y] = 1
        for x, y in result2[2].get():
            image[x][y] = 1
        for x, y in result2[3].get():
            image[x][y] = 1

        if (result1[0].get().__len__() + result1[1].get().__len__() + result1[2].get().__len__() + result1[
            3].get().__len__()) == 0 and \
                        (result2[0].get().__len__() + result2[1].get().__len__() + result2[2].get().__len__() + result2[
                            3].get().__len__()) == 0:
            break   
        else:
            result1 = []
            result2 = []
        for i in range(4):
            result1.append(pool1.apply_async(scanner_1, (image, row_pairs[i][0], row_pairs[i][1])))

    return image


def is_neighbour(pair1, pair2, neighbour_range=2):
    """check whether two point(represented by pair) are neighbour or not"""
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

    # plt.matshow(temp_map)
    # plt.show()
    # plt.pause(10)

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
            # find the point closest to point_1 and point_2
            point_j = edge_i[j]
            dist_1 = diagonal_distance(point_j, point_1)
            dist_2 = diagonal_distance(point_j, point_2)
            # try to find the j_th point in i_th edge
            if dist_1 < min_1:
                min_1 = dist_1
                index_1 = [i, j]
            if dist_2 < min_2:
                min_2 = dist_2
                index_2 = [i, j]

    # following process is toooooo clumsy
    # get the node number for the edge that point_1 is closest to
    node_11 = road_graph.connection[index_1[0]][0]
    node_12 = road_graph.connection[index_1[0]][1]
    # for point_2
    node_21 = road_graph.connection[index_2[0]][0]
    node_22 = road_graph.connection[index_2[0]][1]

    i_1 = index_1[0]
    j_1 = index_1[1]
    i_2 = index_2[0]
    j_2 = index_2[1]

    if (node_11 == node_21 and node_12 == node_22) or (node_11 == node_22 and node_12 == node_21):
        path_seg = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1], pe=road_graph.edge_list[i_2][j_2])
        return path_seg, [(node_12, node_11)]
    """
    elif node_11 == node_21:
        path_seg1 = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1])
        path_seg2 = road_graph.pixel_path(node_21, node_22, pe=road_graph.edge_list[i_2][j_2])
        path_seg = path_seg1 + path_seg2
        return path_seg, [(node_12, node_11), (node_21, node_22)]
    elif node_11 == node_22:
        path_seg1 = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1])
        path_seg2 = road_graph.pixel_path(node_22, node_21, pe=road_graph.edge_list[i_2][j_2])
        path_seg = path_seg1 + path_seg2
        return path_seg, [(node_12, node_11), (node_22, node_21)]
    elif node_12 == node_21:
        path_seg1 = road_graph.pixel_path(node_11, node_12, ps=road_graph.edge_list[i_1][j_1])
        path_seg2 = road_graph.pixel_path(node_21, node_22, pe=road_graph.edge_list[i_2][j_2])
        path_seg = path_seg1 + path_seg2
        return path_seg, [(node_11, node_12), (node_21, node_22)]
    elif node_12 == node_22:
        path_seg1 = road_graph.pixel_path(node_11, node_12, ps=road_graph.edge_list[i_1][j_1])
        path_seg2 = road_graph.pixel_path(node_22, node_21, pe=road_graph.edge_list[i_2][j_2])
        path_seg = path_seg1 + path_seg2
        return path_seg, [(node_11, node_12), (node_22, node_21)]
    """

    # can be optimized, not all path needs to be stored and transferred
    path_1, ind1 = road_graph.a_star_path(node_11, node_21)
    path_2, ind2 = road_graph.a_star_path(node_11, node_22)
    path_3, ind3 = road_graph.a_star_path(node_12, node_21)
    path_4, ind4 = road_graph.a_star_path(node_12, node_22)

    if isinstance(ind1, tuple):
        ind1 = [ind1]
    if isinstance(ind2, tuple):
        ind2 = [ind2]
    if isinstance(ind3, tuple):
        ind3 = [ind3]
    if isinstance(ind4, tuple):
        ind4 = [ind4]
    #
    # l1 = int(diagonal_distance(point_1, road_graph.nodes[node_11]) +
    #          diagonal_distance(point_2, road_graph.nodes[node_21]))
    # l2 = int(diagonal_distance(point_1, road_graph.nodes[node_11]) +
    #          diagonal_distance(point_2, road_graph.nodes[node_22]))
    # l3 = int(diagonal_distance(point_1, road_graph.nodes[node_12]) +
    #          diagonal_distance(point_2, road_graph.nodes[node_21]))
    # l4 = int(diagonal_distance(point_1, road_graph.nodes[node_12]) +
    #          diagonal_distance(point_2, road_graph.nodes[node_22]))

    l1 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_11]))) + \
         abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_21])))
    l2 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_11]))) + \
         abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_22])))
    l3 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_12]))) + \
         abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_21])))
    l4 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_12]))) + \
         abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_22])))

    for it in ind1:
        l1 = l1 + road_graph.get_length(it)
    for it in ind2:
        l2 = l2 + road_graph.get_length(it)
    for it in ind3:
        l3 = l3 + road_graph.get_length(it)
    for it in ind4:
        l4 = l4 + road_graph.get_length(it)

    if l1 < l2 and l1 < l3 and l1 < l4:
        path_seg = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1])
        for it in ind1:
            path_seg += road_graph.pixel_path(it[0], it[1])
        path_seg += road_graph.pixel_path(node_21, node_22, pe=road_graph.edge_list[i_2][j_2])
        return path_seg, ind1
    elif l2 < l1 and l2 < l3 and l2 < l4:
        path_seg = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1])
        for it in ind2:
            path_seg += road_graph.pixel_path(it[0], it[1])
        path_seg += road_graph.pixel_path(node_22, node_21, pe=road_graph.edge_list[i_2][j_2])
        return path_seg, ind2
    elif l3 < l1 and l3 < l2 and l3 < l4:
        path_seg = road_graph.pixel_path(node_11, node_12, ps=road_graph.edge_list[i_1][j_1])
        for it in ind3:
            path_seg += road_graph.pixel_path(it[0], it[1])
        path_seg += road_graph.pixel_path(node_21, node_22, pe=road_graph.edge_list[i_2][j_2])
        return path_seg, ind3
    else:
        path_seg = road_graph.pixel_path(node_11, node_12, ps=road_graph.edge_list[i_1][j_1])
        for it in ind4:
            path_seg += road_graph.pixel_path(it[0], it[1])
        path_seg += road_graph.pixel_path(node_22, node_21, pe=road_graph.edge_list[i_2][j_2])
        return path_seg, ind4


def diagonal_distance(point_a, point_b):
    dx = float(abs(point_a[0] - point_b[0]))
    dy = float(abs(point_a[1] - point_b[1]))
    return math.sqrt(dx*dx + dy*dy)


def mid_point(a1, a2):
    b = np.array(np.zeros(a1.size), dtype=a1.dtype)
    for i in range(a1.size):
        b[i] = (a1[i] + a2[i]) / 2.0
    return b


def pixeltotrue(pi, pj):
	x_max = 120.
	y_max = 120.
	
	return [ (2.4/x_max)*(x_max - 1 - pj) - 0.44, -(2.4/y_max) * pi + 0.2 ]


if __name__ == "__main__":

	# get an image representing the obstacles and walls
	img_list = ["maze_2017.png", "maze2_more.png", "maze2_less.png"]
	img_path = "/home/ras16/catkin_ws/src/path_planning/scripts/"

	t_start = time.time()
	original = cv2.imread(img_path + img_list[0])

	height, width = original.shape[:2]

	# resize the image to get lower resolution
	# this makes computational cost decrease much
	# original = cv2.resize(original, (int(0.2 * width), int(0.2 * height)), interpolation=cv2.INTER_CUBIC)
	original = cv2.resize(original, (120, 120), interpolation=cv2.INTER_CUBIC)
	print "image height" + str(120) + "image width" + str(120)

	gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
	# use only binary map
	# with probabilistic occupancy, the threshold of this should be set carefully
	ret3, my_map_original = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	t_end = time.time()
	print "time for preparation: " + str(t_end - t_start)

	t_start = time.time()
	# do erosion first, and do thinning
	# this Zhang Suen thinning guarantees that skeleton will not be cut at any point
	thinning = zs_thinning(my_map_original)
	t_end = time.time()
	print "time for thinning: " + str(t_end - t_start)

	t_start = time.time()
	joint_count = 0

	node_list = []

	# find the joint node of branching point
	th, tw = thinning.shape[:2]

	for pi in range(1, th - 1):
		for pj in range(1, tw - 1):
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
				
				# point with at least 3 neighbor points who are not neighbor to each other
				if (white_to_black(p2, p3) + white_to_black(p3, p4) +
						white_to_black(p4, p5) + white_to_black(p5, p6) +
						white_to_black(p6, p7) + white_to_black(p7, p8) +
						white_to_black(p8, p9) + white_to_black(p9, p2) >= 3):
					joint_count = joint_count + 1
					thinning[pi][pj] = 150
					node_list.append((pi, pj))
				# point at the end of an edge
				if black_n == 1:
					thinning[pi][pj] = 150
					joint_count = joint_count + 1
					node_list.append((pi, pj))

	voronoi_graph = RoadGraph(node_list)
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
	res = res + 1 # to cope with the problem of old version regionprops
	props = regionprops(res)

	print "label number " + str(label_num) + "  " + str(len(props))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.matshow(res)

	num_n = 0
	# show the ndoe with their number in node_list
	for n_i, n_j in node_list:
	 	ax.text(n_j, n_i, str(num_n), color='red', fontsize=12)
	 	num_n += 1

	# to show the walls in original map which is removed from the map
	for pi in range(1, th - 1):
		for pj in range(1, tw - 1):
			if my_map_original[pi][pj] == 0:
				res[pi][pj] = label_num + 2

	# in edge list, we have label_num edges, made up by a list of coordinate pairs
	edge_list = dict()
	for l_i in range(label_num):
		temp_list = props[l_i].coords

		ordered_list = []
		swap_list = temp_list.tolist()
		seq = list()
		# get sequences that in each one of them the pixels are in order
		for i in range(len(swap_list)-1):
			if len(seq) == 0:
				seq.append(swap_list[i])
			if is_neighbour(swap_list[i], swap_list[i+1], neighbour_range=1):
				seq.append(swap_list[i+1])
				continue
			else:
				ordered_list.append(seq)
				seq = list()
				seq.append(swap_list[i+1])
		ordered_list.append(seq)

		swap_list = list()
		# following part concatentes the sequences in order
		while ordered_list.__len__() > 0:
			for item_i in xrange(len(ordered_list) - 1, -1, -1):
				item = ordered_list[item_i]
				if len(swap_list) == 0:
					swap_list = item
					del ordered_list[item_i]
				else:
					if is_neighbour(swap_list[0], item[0]):
						swap_list = item[::-1] + swap_list
						del ordered_list[item_i]
					elif is_neighbour(swap_list[0], item[-1]):
						swap_list = item + swap_list
						del ordered_list[item_i]
					elif is_neighbour(swap_list[-1], item[0]):
						swap_list = swap_list + item
						del ordered_list[item_i]
					elif is_neighbour(swap_list[-1], item[-1]):
						swap_list = swap_list + item[::-1]
						del ordered_list[item_i]
		temp_list = swap_list
		temp_pair = [-1, -1]
		# append the edges and node pair to the graph object
		for node_i in range(len(node_list)):
			if is_neighbour(temp_list[0], node_list[node_i]):
				temp_pair[0] = node_i
				temp_list.insert(0, list(node_list[node_i]))
			elif is_neighbour(temp_list[-1], node_list[node_i]):
				temp_pair[1] = node_i
				temp_list.append(list(node_list[node_i]))

		# if the edge does not link two different nodes
		# then it must be a loop since it is detected as an edge
		if len(temp_pair) == 1:
			temp_pair.append(temp_pair[0])

		edge_list[(temp_pair[0], temp_pair[1])] = temp_list
		voronoi_graph.add_edge((temp_pair[0], temp_pair[1]), temp_list)

	t_end = time.time()
	print "time for graph making: " + str(t_end - t_start)

# ------------------------prepration work done-----------------------------

	pub = rospy.Publisher('guider', Float32MultiArray, queue_size=1)
	rospy.init_node('path_planning', anonymous=True)
	rate = rospy.Rate(5)
	
	"""let's try find path between any two point!"""
	start_point = (80, 9)
	end_point = (60, 80)

	ax.text(start_point[1], start_point[0], 'S', color='red', fontsize=12)
	ax.text(end_point[1], end_point[0], 'E', color='red', fontsize=12)

	path2, pd2 = path_any_point(voronoi_graph, my_map_original, start_point, end_point)

	for pdi in range(len(path2)):
	 	ax.text(path2[pdi][1], path2[pdi][0], str(pdi), color='white', fontsize=14)

	plt.show()
	plt.pause(10)

	"""
	path_msg = Float32MultiArray()
	for waypoint in path2:
		truepoint = pixeltotrue(waypoint[1], waypoint[0])
		path_msg.data.append(truepoint[0])
		path_msg.data.append(truepoint[1])

	while not rospy.is_shutdown():
		pub.publish(path_msg)
		print "message sent"
		print path2
		rate.sleep()
	"""
	
	
	"""in pd2, the first and last element are just cutting point into some edge"""
	# if len(pd2) != 0:
	# 	print pd2

	# plt.show()
	# plt.pause(60)

	# print thinning[20]
	
