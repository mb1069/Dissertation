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

    ax1.scatter([p[0] for p in startdata], [p[1] for p in startdata], s=3, c='r', marker='x')
    ax1.scatter([p[0] for p in finaldata], [p[1] for p in finaldata], s=3, c='g', marker='x')
    ax1.scatter([p[0] for p in refpoints], [p[1] for p in refpoints], s=4, c='b', marker='x')

    plt.show()

def graph_gen(refpoints, pop, target):
    plt.ion()
    fig = plt.figure("X/Y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    ax1 = fig.add_subplot(111)

    pop_series = ax1.scatter([p[0] for p in pop], [p[1] for p in pop], s=3, c='r', marker='x')
    ax1.scatter(target[0], target[1], s=3, c='g', marker='x')
    ax1.scatter([p[0] for p in refpoints], [p[1] for p in refpoints], s=4, c='b', marker='x')  
    plt.show()
    return fig, pop_series

def update_series(fig, series, newData):
    fig.canvas.draw_idle()
    plt.pause(0.1)
    series.set_offsets([(x[0], x[1]) for x in newData])
    fig.canvas.draw()

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

def evaluate_solution(posX, posY, rot, targetX, targetY, targetRot):
    solution = np.array((posX, posY))
    target = np.array((targetX, targetY))
    return (euclid_distance(solution, target), rot/targetRot)


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

def total_sum(set1, set2, banded=False):
    h = 0
    for p1 in set1:
        shortest = float('inf')
        for p2 in set2:
            dist = euclid_distance(p1, p2)
            if dist<shortest:
                shortest = dist
        h += apply_band(shortest) if banded else shortest
    return h

def apply_band(dist):
    if dist<=1: return dist
    if 1<dist<=3: return dist * 2
    if 3<dist<=5: return dist * 4
    if 5<dist<=7: return dist * 8
    if 7<dist: return dist * 16


def subsample(points, tolerance):
    newpoints = []
    for p1 in points:
        if not containsSimilar(newpoints, p1, tolerance):
            newpoints.append(p1)
    return newpoints


def containsSimilar(set, p, tolerance):
    for p2 in set:
        if abs(p2[0]-p[0])<=tolerance and abs(p2[1]-p[1])<tolerance:
            return True
    return False


def save_data(row, filename):
    line = ",".join(map(str, row))
    fd = open(filename, 'a+')
    fd.write(line)
    fd.close()

def initPop(POP, map, constr):
    minX = min([x[0] for x in map])
    minY = min([x[1] for x in map])    
    maxX = max([x[0] for x in map])
    maxY = max([x[1] for x in map])
    dimX = float(maxX - minX)
    dimY = float(maxY - minY)
    POP = float(POP)
    Xpoints = int(math.sqrt(((dimX*POP)/dimY)+(math.pow(dimX-dimY, 2)/(4*(dimY**2))))-((dimX-dimY)/(2*dimY)))
    Ypoints = int(POP/(Xpoints))

    Xstep = dimX/(Xpoints-1)
    Ystep = dimY/(Ypoints-1)
    pop = []
    for x in range(Xpoints):
        for y in range(Ypoints):
            for rot in range(0,4):
                ind = constr([(x*Xstep) + minX, (y*Ystep)+ minY, 0])
                pop.append(ind)

    # # plt.ion()
    # fig = plt.figure("X/Y")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax1 = fig.add_subplot(111)

    # pop_series = ax1.scatter([p[0] for p in pop], [p[1] for p in pop], s=3, c='r', marker='x')
    # # ax1.scatter(target[0], target[1], s=3, c='g', marker='x')
    # ax1.scatter([p[0] for p in map], [p[1] for p in map], s=4, c='b', marker='x')  
    # plt.show()
    return pop

