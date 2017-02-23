import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


def copy_scan(scan):
    return copy.deepcopy(scan)

def graph_results(refpoints, errorscan, transformation):
    startdata = copy.deepcopy(errorscan)
    finaldata = applytuple(errorscan, *transformation)
    fig = plt.figure("X/Y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    ax1 = fig.add_subplot(111)

    ax1.scatter([p[0] for p in startdata], [p[1] for p in startdata], s=2, c='r', marker='x')
    ax1.scatter([p[0] for p in finaldata], [p[1] for p in finaldata], s=2, c='g', marker='x')
    ax1.scatter([p[0] for p in refpoints], [p[1] for p in refpoints], s=2, c='b', marker='x')

    plt.show()


def applytuple(scan_points, xErr, yErr, rotErr):
    points = []
    for coord in scan_points:
        x = coord[0]
        y = coord[1]
        # Applying rotation
        x, y = applyrotation(x, y, rotErr)
        x, y = applytranslation(x, y, xErr, yErr)
        points.append(np.array((x, y)))
    return points

def polar2origincartesian(scan, dist, ang):
    dist = float(dist)
    ang = float(ang)
    return np.array((scan.posx + dist * math.cos(ang+scan.rot), scan.posy + dist * math.sin(ang+scan.rot)))


def polar2cartesian(dist, ang):
    dist = float(dist)
    ang = float(ang)
    return np.array((dist * math.cos(ang), dist * math.sin(ang)))


def applyrotation(x, y, rotErr):
    return (x * math.cos(rotErr)) - y * (math.sin(rotErr)), (x * math.sin(rotErr)) + y * (math.cos(rotErr))


def applytranslation(x, y, xErr, yErr):
    return x + xErr, y + yErr


def euclid_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def hausdorff(set1, set2):
    h = 0
    for p1 in set1:
        shortest = float('inf')
        for p2 in set2:
            dist = euclid_distance(p1, p2)
            if dist < shortest:
                shortest = dist
        if shortest > h:
            h = shortest
    return h

def total_sum(set1, set2):
    h = 0
    for p1 in set1:
        shortest = float('inf')
        for p2 in set2:
            dist = euclid_distance(p1, p2)
            if dist<shortest:
                shortest = dist
        h += shortest
    return h
