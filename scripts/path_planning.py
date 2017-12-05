#!/usr/bin/env python
import roslib
import sys
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import time
import math
import multiprocessing
from random import randint


class FindAllCycles:
    def __init__(self, graph_input):
        """graph input should be a list of pair"""
        self.graph = graph_input
        self.cycles = list()

    def run_it(self):
        result = list()
        for edge in self.graph:
            for node in edge:
                self.find_new_cycles([node])
        return self.cycles

    def find_new_cycles(self, path):
        start_node = path[0]
        next_node = None
        subnode = []
        # visit each edge and each node of each edge
        for edge in self.graph:
            node1, node2 = edge
            if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
            if not (next_node in path):
                # neighbor node not on path yet
                subnode = [next_node]
                subnode.extend(path)
                # explore extended path
                self.find_new_cycles(subnode)
            elif len(path) > 2 and next_node == path[-1]:
                # cycle found
                p = self.rotate_to_smallest(path)
                inv = self.rotate_to_smallest(p[::-1])
                if not (p in self.cycles) and not (inv in self.cycles):
                    self.cycles.append(p)

    # rotate cycle path such that it begins with the smallest node
    @staticmethod
    def rotate_to_smallest(path):
        n = path.index(min(path))
        return path[n:] + path[:n]


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
        self.edge_dict[edge_key] = (edge_value.__len__(), self.edge_list.__len__() - 1)
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
        euclidean = int(math.sqrt(math.pow(abs(x1 - x2), 2) + math.pow(abs(y1 - y2), 2)) / 1.414)
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
            return (0, 0)

        for con in self.connection:
            if con == tuple([node_1, node_2]) or con == tuple([node_2, node_1]):
                print "Warning: directly connected nodes!"
                return (node_1, node_2)

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
        path_index = list()
        pair_temp = (node_q.index, node_end.index)
        path_index.append(pair_temp)

        current_node = node_q
        while current_node.parent != -1:
            for node_left in closed_nodes:
                if node_left.index == current_node.parent:
                    pair_temp = (node_left.index, current_node.index)
                    path_index.insert(0, pair_temp)
                    current_node = node_left
                    break
        return path_index

    def pixel_path(self, p1, p_2, ps=None, pe=None, interval=11):
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
            temp_edge = temp_edge[cut1:(cut2 + 1)]
        else:
            temp_edge = temp_edge[cut2:(cut1 + 1)]
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
        print "Warning: due to some reason, end node is not in the pixel path"
        return pixel_path


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

    p2 = int(img[i - 1][j])
    p3 = int(img[i - 1][j + 1])
    p4 = int(img[i][j + 1])
    p5 = int(img[i + 1][j + 1])
    p6 = int(img[i + 1][j])
    p7 = int(img[i + 1][j - 1])
    p8 = int(img[i][j - 1])
    p9 = int(img[i - 1][j - 1])

    b = int(8 - (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9))

    # print b

    if b < 2 or b > 6:
        return False

    a = int((p2 and p3 == 0) + (p3 and p4 == 0) + (p4 and p5 == 0) + (p5 and p6 == 0) + (p6 and p7 == 0) + (
    p7 and p8 == 0) + (p8 and p9 == 0) + (p9 and p2 == 0))

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
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.erode(image, kernel1, iterations=5)

    image = cv2.bitwise_not(image)

    row, col = image.shape[:2]

    # plt.matshow(image, 1)
    # plt.matshow(img, 2)
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


def diagonal_distance(point_a, point_b):
    dx = float(abs(point_a[0] - point_b[0]))
    dy = float(abs(point_a[1] - point_b[1]))
    return math.sqrt(dx * dx + dy * dy)


def mid_point(a1, a2):
    b = np.array(np.zeros(a1.size), dtype=a1.dtype)
    for i in range(a1.size):
        b[i] = (a1[i] + a2[i]) / 2.0
    return b

class PathPlanner:
    def __init__(self, init_map):
        self.global_map = init_map
        self.graph = self.update_graph(init_map)

    def update_map(self, new_map):
        self.global_map = new_map
        self.graph = self.update_graph(new_map)

    def update_graph(self, current_map):

        # global ax
        
        t_start = time.time()
        thinning = zs_thinning(current_map)
        t_end = time.time()
        print "time for thinning: " + str(t_end - t_start)
        
        joint_count = 0


        node_list = []
        t_start = time.time()
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

        # to show the node with a larger square dot
        # in labelling of connected components,
        # it is also used as edge cutting
        for pi, pj in node_list:
            for ia in range(-1, 2):
                for ja in range(-1, 2):
                    thinning[pi + ia][pj + ja] = 1

        # to label all the connected component
        res, label_num = label(thinning, neighbors=8, background=1, return_num=True)
        res = res + 1  # to cope with the problem of old version regionprops
        props = regionprops(res)

        print "label number " + str(label_num) + "  " + str(len(props))

        # ax.matshow(res)

        num_n = 0
        # show the ndoe with their number in node_list

        # for n_i, n_j in node_list:
        # 	ax.text(n_j, n_i, str(num_n), color='red', fontsize=12)
        #	num_n += 1

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
            for i in range(len(swap_list) - 1):
                if len(seq) == 0:
                    seq.append(swap_list[i])
                if is_neighbour(swap_list[i], swap_list[i + 1], neighbour_range=1):
                    seq.append(swap_list[i + 1])
                    continue
                else:
                    ordered_list.append(seq)
                    seq = list()
                    seq.append(swap_list[i + 1])
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
            if len(temp_list) < 10:
                continue
            # append the edges and node pair to the graph object
            for node_i in range(len(node_list)):
                if is_neighbour(temp_list[0], node_list[node_i]) or \
                        is_neighbour(temp_list[1], node_list[node_i]) or \
                        is_neighbour(temp_list[2], node_list[node_i]):
                    temp_pair[0] = node_i
                    temp_list.insert(0, list(node_list[node_i]))
                elif is_neighbour(temp_list[-1], node_list[node_i]) or \
                        is_neighbour(temp_list[-2], node_list[node_i]) or \
                        is_neighbour(temp_list[-3], node_list[node_i]):
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

        return voronoi_graph

    def path_any_point(self, point_1, point_2, go_edge=False):
        """find path between any two point in map"""
        road_graph = self.graph
        
        if go_edge:
			path_seg = road_graph.pixel_path(point_1, point_2)
			return path_seg
			
        temp_map = self.global_map.copy()

        # this to a degree can represent the expansion of walls and obstacles
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        temp_map = cv2.erode(temp_map, kernel1, iterations=5)

        # plt.matshow(temp_map)
        # plt.show()
        # plt.pause(10)

        if temp_map[point_2[0]][point_2[1]] == 0:
            print("invalid starting or ending position! You are colliding the wall there!")

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

        # print node_11, node_12, node_21, node_22

        i_1 = index_1[0]
        j_1 = index_1[1]
        i_2 = index_2[0]
        j_2 = index_2[1]

        if (node_11 == node_21 and node_12 == node_22) or (node_11 == node_22 and node_12 == node_21):
            path_seg = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1],
                                             pe=road_graph.edge_list[i_2][j_2])
            return path_seg, [(node_12, node_11)]

        # can be optimized, not all path needs to be stored and transferred
        ind1 = road_graph.a_star_path(node_11, node_21)
        ind2 = road_graph.a_star_path(node_11, node_22)
        ind3 = road_graph.a_star_path(node_12, node_21)
        ind4 = road_graph.a_star_path(node_12, node_22)

        if isinstance(ind1, tuple):
            ind1 = [ind1]
        if isinstance(ind2, tuple):
            ind2 = [ind2]
        if isinstance(ind3, tuple):
            ind3 = [ind3]
        if isinstance(ind4, tuple):
            ind4 = [ind4]

        l1 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_11]))) + \
             abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_21])))
        l2 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_11]))) + \
             abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_22])))
        l3 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_12]))) + \
             abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_21])))
        l4 = abs(index_1[1] - road_graph.edge_list[index_1[0]].index(list(road_graph.nodes[node_12]))) + \
             abs(index_2[1] - road_graph.edge_list[index_2[0]].index(list(road_graph.nodes[node_22])))

        #print i_1, j_1, i_2, j_2
        #print ind1, ind2, ind3, ind4
        #print l1, l2, l3, l4

        for it in ind1:
            if not it == (0, 0):
                l1 = l1 + road_graph.get_length(it)
                print road_graph.get_length(it)
        for it in ind2:
            if not it == (0, 0):
                l2 = l2 + road_graph.get_length(it)
                print road_graph.get_length(it)
        for it in ind3:
            if not it == (0, 0):
                l3 = l3 + road_graph.get_length(it)
                print road_graph.get_length(it)
        for it in ind4:
            if not it == (0, 0):
                l4 = l4 + road_graph.get_length(it)
                print road_graph.get_length(it)

        if l1 < l2 and l1 < l3 and l1 < l4:
            path_seg = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1])
            for it in ind1:
                if not it == (0, 0):
                    path_seg += road_graph.pixel_path(it[0], it[1])
            path_seg += road_graph.pixel_path(node_21, node_22, pe=road_graph.edge_list[i_2][j_2])
            return path_seg, ind1
        elif l2 < l1 and l2 < l3 and l2 < l4:
            path_seg = road_graph.pixel_path(node_12, node_11, ps=road_graph.edge_list[i_1][j_1])
            for it in ind2:
                if not it == (0, 0):
                    path_seg += road_graph.pixel_path(it[0], it[1])
            path_seg += road_graph.pixel_path(node_22, node_21, pe=road_graph.edge_list[i_2][j_2])
            return path_seg, ind2
        elif l3 < l1 and l3 < l2 and l3 < l4:
            path_seg = road_graph.pixel_path(node_11, node_12, ps=road_graph.edge_list[i_1][j_1])
            for it in ind3:
                if not it == (0, 0):
                    path_seg += road_graph.pixel_path(it[0], it[1])
            path_seg += road_graph.pixel_path(node_21, node_22, pe=road_graph.edge_list[i_2][j_2])
            return path_seg, ind3
        else:
            path_seg = road_graph.pixel_path(node_11, node_12, ps=road_graph.edge_list[i_1][j_1])
            for it in ind4:
                if not it == (0, 0):
                    path_seg += road_graph.pixel_path(it[0], it[1])
            path_seg += road_graph.pixel_path(node_22, node_21, pe=road_graph.edge_list[i_2][j_2])
            return path_seg, ind4

    def simple_exploration(self):
        edges_ = self.graph.connection

        explorer = FindAllCycles(edges_)
        cycles = explorer.run_it()

        cycles_edges = list()
        for cy in cycles:
            cy_temp = []
            for node_num_ in range(len(cy) - 1):
                cy_temp.append((cy[node_num_], cy[node_num_ + 1]))
            cy_temp.append((cy[len(cy) - 1], cy[0]))
            cycles_edges.append(cy_temp)

        max_length = 0
        cycle_index = 0
        for c_i in range(len(cycles_edges)):
            cycle_length = 0
            for e_i in cycles_edges[c_i]:
                cycle_length += self.graph.get_length(e_i)
            if cycle_length > max_length:
                max_length = cycle_length
                cycle_index = c_i

        node_start = 7
        if node_start in cycles[cycle_index]:
            while cycles[cycle_index][0] != node_start:
                ttt = cycles[cycle_index].pop(0)
                cycles[cycle_index].append(ttt)
            cycles[cycle_index].append(node_start)
        else:
            min_length = 99999
            min_node = 0
            for node_i_ in cycles[cycle_index]:
                beginning_road = 0
                indexes = self.graph.a_star_path(node_i_, node_start)
                if isinstance(indexes, tuple):
                    indexes = [indexes]
                for edge_i_ in indexes:
                    if not edge_i_ == (0, 0):
                        beginning_road += self.graph.get_length(edge_i_)
                # print indexes
                # print "road _length: " + str(beginning_road) + " node " + str(node_i_)
                if beginning_road < min_length:
                    min_length = beginning_road
                    min_node = node_i_

            while cycles[cycle_index][0] != min_node:
                ttt = cycles[cycle_index].pop(0)
                cycles[cycle_index].append(ttt)
            cycles[cycle_index].insert(0, node_start)
            cycles[cycle_index].append(min_node)
            cycles[cycle_index].append(node_start)

        return cycles[cycle_index]


def targetCallback(msg):
    global target_point
    global planner_status
    if planner_status == 3 or planner_status == 1:
		target_t = (msg.points[0].x, msg.points[0].y)
		if target_t != target_point:
			target_point = target_t
			print("target changed! " + str(target_point))
			planner_status = 3


def poseCallback(msg):
    global current_path
    global next_target_num
    global current_position
    global time_start
    global planner_1
    global planner_status
    global update_position
    global target_point
    global history
    global starting_point
    global wall_time
    global is_new_
    current_position = (msg.pose.position.x, msg.pose.position.y)
    if not is_new_:
		if len(current_path) > 0 and next_target_num < len(current_path):
			dx = current_path[next_target_num][0] - msg.pose.position.x
			dy = current_path[next_target_num][1] - msg.pose.position.y
			distance = math.sqrt(dx * dx + dy * dy)
			if distance <= 0.15:
				next_target_num += 1
				time_start = rospy.get_time()

		# if planner_status == 1 and time_start != 0.0 and rospy.get_time() - time_start > 15.0:
		# what should it do here? in second round we should not stuck any more !
		
    else: 
		pixel_current_position = truetopixel(current_position[0],current_position[1])
		
		if (target_point == (0, 0)):
			starting_point = (105, 10)
			max_dist = 0
			for rand_node in history:
				dist_temp = diagonal_distance(rand_node, pixel_current_position)
				if dist_temp > max_dist:
					max_dist = dist_temp
					max_node = rand_node
			"""
			rand_num = randint(0, len(history)-1)
			rand_node = history[rand_num]
			print("rand_node" + str(rand_node))
			while True:
				if diagonal_distance(rand_node, pixel_current_position) < 10:
					del history[rand_num]
					rand_num = randint(0, len(history)-1)
					rand_node = history[rand_num]
					print("rand_node" + str(rand_node))
				else:
					print("rand_node" + str(rand_node))
					break
			"""
			
			target_point = pixeltotrue(max_node[1], max_node[0])
			print("chosen farthest point " + str(max_node))
			planner_status = 3
		
		if rospy.get_time() - wall_time > 240:
			print("going home!!!!!!!!!!--------------------------------------------------")
			target_point = pixeltotrue(starting_point[1], starting_point[0])
			planner_status = 3
			return
		
		for nd_i in xrange(len(history) - 1, -1, -1):
			if diagonal_distance(history[nd_i], pixel_current_position) < 12:
				print("WE HAVE MET: " + str(history[nd_i]) + "-----------------------------------")
				del history[nd_i]
		
		if truedistance(target_point, current_position) < 0.12  and len(history) > 0:
			print("reach target! resetting target")
			max_dist = 0
			for rand_node in history:
				dist_temp = diagonal_distance(rand_node, pixel_current_position)
				if dist_temp > max_dist:
					max_dist = dist_temp
					max_node = rand_node
			target_point = pixeltotrue(max_node[1], max_node[0])
			planner_status = 3
			
		if truedistance(current_position, update_position) > 0.3:
			time_start = rospy.get_time()
			update_position = current_position
			print("moving!")
		else:
			if rospy.get_time() - time_start > 12 and len(history) > 0:
				print("not moving! resetting target")
				
			    # reading the map or not
				"""
				new_maze = cv2.imread("/home/ras26/catkin_ws/src/ras_map/map_image/init_map.png")
				new_maze = cv2.bitwise_not(new_maze)
				# new_maze = cv2.resize(new_maze, (120, 120), interpolation=cv2.INTER_CUBIC)
				new_maze = cv2.cvtColor(new_maze, cv2.COLOR_RGB2GRAY)
				ret3, new_maze = cv2.threshold(new_maze, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				#plt.hist(new_maze.ravel(),256,[0,256]) 
				#plt.show()
				
				# plt.matshow(new_maze)
				# plt.pause(10)
				planner_1.update_map(new_maze)
				
				print("updated map, doing new plan")
				"""

				max_dist = 0
				for rand_node in history:
					if diagonal_distance(rand_node, truetopixel(target_point[0], target_point[1])) < 20:
						continue
					dist_temp = diagonal_distance(rand_node, pixel_current_position)
					if dist_temp > max_dist:
						max_dist = dist_temp
						max_node = rand_node
				if max_dist == 0:
					target_point = starting_point
				else:
					target_point = pixeltotrue(max_node[1], max_node[0])

				time_start = rospy.get_time()
				update_position = current_position
				planner_status = 3
			
		
def truedistance(point1, point2):
	delta0 = point1[0] - point2[0]
	delta1 = point1[1] - point2[1]
	return math.sqrt(delta0*delta0 + delta1*delta1)		
    

def pixeltotrue(pi, pj):
    x_max = 120.
    y_max = 120.

    return [(2.4 / y_max) * pi, (2.4 / x_max) * (x_max - 1 - pj)]


def truetopixel(px, py):
    x_max = 120.
    y_max = 120.

    return [int(x_max - 1 - (x_max/2.4)*py), int((y_max/2.4)*px)]


if __name__ == "__main__":

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # get an image representing the obstacles and walls
    img_path = "/home/ras26/catkin_ws/src/ras_map/map_image/init_map.png"

    t_start = time.time()
    original = cv2.imread(img_path)
    
    original = cv2.bitwise_not(original)

    print original.shape[:2]

    # resize the image to get lower resolution, this makes computational cost decrease much
    # original = cv2.resize(original, (120, 120), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    
    # use only binary map
    # with probabilistic occupancy, the threshold of this should be set carefully
    ret3, my_map_original = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    planner_1 = PathPlanner(my_map_original)

    # ------------------------prepration work done-----------------------------

    rospy.init_node('path_planning', anonymous=False)
    
    # 0 for waiting, 1 for sending out path
    planner_status = rospy.get_param('planner_status', 3)
    is_new_ = rospy.get_param('is_explore', True)
    
    pub = rospy.Publisher('guider', Float32MultiArray, queue_size=1)
    sub = rospy.Subscriber("target_point", Polygon, targetCallback)
    if not is_new_:
		sub_pose = rospy.Subscriber("/localization/pose", PoseStamped, poseCallback)
    """do simple exploration"""
    current_position = (0, 0)
    target_point = (0, 0)
    update_position = (0, 0)
    starting_point = (0, 0)
    
    """create nodes list to travel"""
    history = planner_1.graph.nodes[:]
	
    for nd_i in xrange(len(history) - 1, -1, -1):
		if diagonal_distance(history[nd_i], [105, 10]) < 12:
			del history[nd_i]
			
    for edge_noi in planner_1.graph.edge_list:
		half_length = int(len(edge_noi) / 2)
		history.append(tuple(edge_noi[half_length]))

    print("this is the edge list " + str(history))
    
    command_num = 0
    
    path_msg = Float32MultiArray()
    
    time_start = rospy.get_time()
    wall_time = rospy.get_time()
    next_target_num = 0
    count = 1
    
    current_path = []
    
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
		print target_point, current_position
		try:
			msg = rospy.wait_for_message("/localization/pose", PoseStamped)
			poseCallback(msg)
			
			if planner_status == 2:
				print("exploration------------------------------------------------")
				test_path = planner_1.simple_exploration()
				explore_path = []
				for ii in range(len(test_path)-1):
					explore_path += planner_1.path_any_point(test_path[ii], test_path[ii+1], go_edge=True)
				# explore_path.reverse()
				print explore_path
				
				for num_i in xrange(len(explore_path) - 1, 0, -1):
					if diagonal_distance(explore_path[num_i], explore_path[num_i-1]) <= 5:
						del explore_path[num_i-1]
				current_path = []
				path_msg.data = []
				for pdi in range(2, len(explore_path)):
					truepoint = pixeltotrue(explore_path[pdi][1], explore_path[pdi][0])
					path_msg.data.append(truepoint[0])
					path_msg.data.append(truepoint[1])
					current_path.append((truepoint[0], truepoint[1]))
				path_msg.data.append(command_num)
				command_num = 1 - command_num
				next_target_num = 0
				print path_msg.data
				time_start = rospy.get_time()
				
			elif planner_status == 3:
				if current_position != (0, 0) and target_point != (0, 0):
					print("one target running-----------------------------------------------------")
					print(truetopixel(target_point[0], target_point[1]))
					path_temp, pd_temp = planner_1.path_any_point(truetopixel(current_position[0], current_position[1]),
					truetopixel(target_point[0], target_point[1]))
					for num_i in xrange(len([path_temp]) - 1, 0, -1):
						if diagonal_distance(path_temp[num_i], path_temp[num_i-1]) <= 5:
							del path_temp[num_i-1]
							
					current_path = []
					x_dots = []
					y_dots = []
					path_msg.data = []
					for pdi in range(1, len(path_temp)):
						truepoint = pixeltotrue(path_temp[pdi][1], path_temp[pdi][0])
						path_msg.data.append(truepoint[0])
						path_msg.data.append(truepoint[1])
						x_dots.append(path_temp[pdi][0])
						y_dots.append(path_temp[pdi][1])
						current_path.append((truepoint[0], truepoint[1]))
					path_msg.data.append(command_num)
					command_num = 1 - command_num
					planner_status = 1
					# plt.plot(x_dots, y_dots)
					# plt.show()
					next_target_num = 0
					print path_msg.data
					time_start = rospy.get_time()
					
			if planner_status == 1:
				#print("sending message")
				for times in range(2):
					pub.publish(path_msg)
					#print "message sent " + str(count)
					count += 1
					#print path_msg.data
					rate.sleep()
				planner_status = 1
				print("waiting for pose")
		except KeyboardInterrupt:
			print("stoping node")
			break
		
		rate.sleep()
        
